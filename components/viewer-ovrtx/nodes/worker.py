"""OVRT render worker and embedded browser surface.

This module deliberately has no Blacknode imports.  It is launched in a child
process so the optional native renderer has an explicit lifecycle boundary.
"""
from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


RENDER_PRODUCT = "/BlacknodeOVRT/Render/Viewport"
STREAM_TO_STDOUT = False
PROTOCOL_STREAM: Any = None

_VISER_HDRI_ASSETS = {
    "apartment": "lebombo_1k.jpg",
    "city": "potsdamer_platz_1k.jpg",
    "dawn": "kiara_1_dawn_1k.jpg",
    "forest": "forest_slope_1k.jpg",
    "lobby": "st_fagans_interior_1k.jpg",
    "night": "dikhololo_night_1k.jpg",
    "park": "rooitou_park_1k.jpg",
    "studio": "studio_small_03_1k.jpg",
    "sunset": "venice_sunset_1k.jpg",
    "warehouse": "empty_warehouse_01_1k.jpg",
}


def _environment_intensity(environment: dict[str, Any]) -> float:
    raw = environment.get("intensity", 1.0)
    return max(0.0, min(100.0, float(1.0 if raw is None else raw)))


def _environment_texture_path(environment: dict[str, Any]) -> str:
    """Resolve a custom map or the matching Viser preset to a renderer texture."""
    if not bool(environment.get("hdri_enabled", True)):
        return ""
    custom = str(environment.get("hdri_path") or "").strip()
    if custom:
        path = Path(custom).expanduser().resolve()
        return str(path) if path.is_file() else ""
    preset = str(environment.get("hdri") or "none").strip().lower()
    filename = _VISER_HDRI_ASSETS.get(preset)
    if not filename:
        return ""
    try:
        spec = importlib.util.find_spec("viser")
    except (ImportError, ValueError):
        spec = None
    roots = list(spec.submodule_search_locations or []) if spec is not None else []
    if spec is not None and spec.origin:
        roots.append(str(Path(spec.origin).parent))
    for root in dict.fromkeys(roots):
        candidate = Path(root) / "client" / "src" / "assets" / filename
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def _distant_light(environment: dict[str, Any]) -> dict[str, Any]:
    configured = dict(environment.get("distant_light") or {})
    rotation = [
        float(value)
        for value in list(configured.get("rotation_deg") or [-35.0, 25.0, -25.0])[:3]
    ]
    rotation.extend([0.0] * (3 - len(rotation)))
    return {
        "enabled": bool(configured.get("enabled", True)),
        "intensity": max(0.0, float(configured.get("intensity", 2500.0))),
        # Browser color inputs and stored hex values are sRGB. USD light color
        # is linear, so convert at the renderer boundary while retaining the
        # original sRGB hex for the editor swatch.
        "color": _srgb_to_linear(
            _hex_color(str(configured.get("color") or "#fff2e0"))
        ),
        "angle_deg": max(0.0, min(180.0, float(configured.get("angle_deg", 4.0)))),
        "rotation_deg": rotation,
    }


def _srgb_to_linear(color: tuple[float, float, float]) -> tuple[float, float, float]:
    def convert(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return tuple(convert(value) for value in color)


def _dome_light_intensity(environment: dict[str, Any]) -> float:
    # OVRTX's reference material-editor scene uses 1000 for an HDRI dome.
    # Keep the color-only fallback gentler; environment intensity controls the
    # dome independently from the persistent distant-light sun.
    base = 1000.0 if _environment_texture_path(environment) else 450.0
    return base * _environment_intensity(environment)


def _environment_dome_usda(path: str, environment: dict[str, Any]) -> str:
    """Author an HDRI dome as population content, preserving its asset type."""
    return f'''#usda 1.0
(
    defaultPrim = "EnvironmentDome"
)

def DomeLight "EnvironmentDome"
{{
    asset inputs:texture:file = {_asset_reference(path)}
    token inputs:texture:format = "latlong"
    float inputs:intensity = {_dome_light_intensity(environment):.6g}
    color3f inputs:color = (1, 1, 1)
    bool inputs:visibleInPrimaryRay = false
}}
'''


def _replace_environment_dome(
    stage: Any,
    handle: int | None,
    path: str,
    environment: dict[str, Any],
    ordinal: int,
) -> int | None:
    """Replace only the authored HDRI dome in an attached OVStage."""
    import ovstage

    if handle is not None:
        ovstage.population.remove_usd(stage, handle)
    next_handle = None
    if path:
        next_handle = ovstage.population.add_usd_reference_from_string(
            stage,
            _environment_dome_usda(path, environment),
            "/BlacknodeOVRT/EnvironmentDome",
        )
    ovstage.population.apply_usd_changes(stage, ordinal)
    return next_handle


def _background_source_type(environment: dict[str, Any]) -> int:
    """Return the RTX background override: default dome or explicit color."""
    return 0 if (
        _environment_texture_path(environment)
        and bool(environment.get("show_background", True))
    ) else 2


def _background_source_tensor(environment: dict[str, Any]) -> Any:
    """Match OVStage's populated uint64 Fabric column for RTX source type."""
    import numpy as np

    return np.asarray([_background_source_type(environment)], dtype=np.uint64)


def _send_event(event: dict[str, Any]) -> None:
    encoded = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
    if PROTOCOL_STREAM is not None:
        PROTOCOL_STREAM.sendall(encoded)
    else:
        print(encoded.decode("utf-8").rstrip(), flush=True)


def _emit_status(phase: str, detail: str) -> None:
    if STREAM_TO_STDOUT:
        _send_event({"type": "status", "phase": phase, "detail": detail})


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.phase = "starting"
        self.error = ""
        self.detail = "Waiting for configuration"
        self.jpeg = b""
        self.frame_number = 0
        self.physics_frame = 0
        self.view_mode = "rgb"
        self.latest_transforms: list[list[float]] | None = None
        self.latest_reference: dict[str, Any] | None = None
        self.camera_actions: list[dict[str, float | str]] = []
        self.visibility_updates: list[dict[str, Any]] = []
        self.stage_updates: list[dict[str, Any]] = []
        self.pick_requests: list[dict[str, float]] = []
        self.selection_requests: list[str] = []
        self.collision_wireframe_count = 0
        self.colliders_visible = False
        self.collision_overlay_pixels = 0
        self.collision_overlay_segments = 0
        self.collision_depth_range = [0.0, 0.0]
        self.stop = False

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "error": self.error,
                "detail": self.detail,
                "frame": self.frame_number,
                "physics_frame": self.physics_frame,
                "view_mode": self.view_mode,
                "has_image": bool(self.jpeg),
            }

    def publish(self, jpeg: bytes) -> None:
        with self.frame_ready:
            self.jpeg = jpeg
            self.frame_number += 1
            self.phase = "streaming"
            self.detail = "OVRT RTX render stream"
            self.frame_ready.notify_all()
        if STREAM_TO_STDOUT:
            _send_event({
                "type": "frame",
                "jpeg": base64.b64encode(jpeg).decode("ascii"),
                "collision_wireframe_count": self.collision_wireframe_count,
                "colliders_visible": self.colliders_visible,
                "collision_overlay_pixels": self.collision_overlay_pixels,
                "collision_overlay_segments": self.collision_overlay_segments,
                "collision_depth_range": self.collision_depth_range,
            })


STATE = SharedState()


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blacknode Newton · OVRT</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#080b11;color:#edf3ff}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#080b11}
#viewport{position:fixed;inset:0;display:grid;place-items:center;user-select:none;cursor:default;touch-action:none}
#viewport.dragging{cursor:grabbing}#stream{width:100%;height:100%;object-fit:contain;display:block;pointer-events:none;-webkit-user-drag:none}
#empty{position:absolute;inset:0;display:grid;place-items:center;background:radial-gradient(circle at 50% 42%,#172131,#080b11 65%)}
#card{max-width:520px;padding:24px;text-align:center}.spinner{width:34px;height:34px;margin:0 auto 15px;border:3px solid #273449;border-top-color:#76b900;border-radius:50%;animation:s 1s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}#phase{font-weight:700;font-size:15px}#detail{margin-top:8px;color:#9ba9bd;font-size:13px;line-height:1.45}
#hud{position:fixed;left:12px;top:12px;padding:8px 10px;border:1px solid #ffffff18;border-radius:8px;background:#080b11bb;backdrop-filter:blur(8px);font-size:11px;line-height:1.5;pointer-events:none}
#hud strong{color:#76b900}#help{position:fixed;right:12px;bottom:12px;padding:7px 9px;border-radius:7px;background:#080b11aa;color:#aab5c5;font-size:10px;pointer-events:none}
#modes{position:fixed;left:50%;top:12px;z-index:4;display:flex;gap:3px;padding:4px;transform:translateX(-50%);border:1px solid #ffffff20;border-radius:9px;background:#080b11dd}
#modes button{padding:5px 8px;border:1px solid transparent;border-radius:6px;background:transparent;color:#aeb9c9;cursor:pointer;font:600 10px Inter,system-ui,sans-serif}#modes button.on{border-color:#76b900aa;background:#76b90022;color:#dfffad}
</style></head><body>
<div id="viewport"><img id="stream" src="/stream.mjpg" alt="OVRT render stream" draggable="false"><div id="empty"><div id="card"><div class="spinner"></div><div id="phase">Starting NVIDIA OVRT</div><div id="detail">The first render initializes and caches RTX shaders.</div></div></div></div>
<div id="hud"><strong>OVRT</strong> · <span id="state">starting</span><br>render <span id="frame">0</span> · physics <span id="physics">0</span></div>
<div id="modes"><button data-mode="rgb" class="on">RGB</button><button data-mode="depth">Depth IR</button><button data-mode="segmentation">Segments</button><button data-mode="detection">Boxes</button><button data-mode="composite">Composite</button></div>
<div id="help">Select: click · Orbit: Alt+left · Pan: Alt+middle · Zoom: Alt+right/wheel</div>
<script>
const viewport=document.querySelector('#viewport'),stream=document.querySelector('#stream'),empty=document.querySelector('#empty');let drag=null,pending=null,raf=0;
async function action(body){try{await fetch('/api/camera',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})}catch{}}
async function pick(e){const r=viewport.getBoundingClientRect(),iw=stream.naturalWidth||r.width,ih=stream.naturalHeight||r.height,k=Math.min(r.width/iw,r.height/ih),w=iw*k,h=ih*k,left=r.left+(r.width-w)/2,top=r.top+(r.height-h)/2,x=(e.clientX-left)/w,y=(e.clientY-top)/h;if(x>=0&&x<=1&&y>=0&&y<=1)try{await fetch('/api/pick',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({x,y})})}catch{}}
function flush(){raf=0;if(!pending)return;const body=pending;pending=null;action(body)}
function queueMove(kind,dx,dy){if(kind==='zoom')pending={action:kind,delta:(pending?.delta||0)-(dx+dy)*Math.SQRT1_2};else if(pending&&pending.action===kind){pending.dx+=dx;pending.dy+=dy}else pending={action:kind,dx,dy};if(!raf)raf=requestAnimationFrame(flush)}
function endDrag(e){if(drag&&!drag.alt&&drag.button===0&&drag.distance<4)pick(e);drag=null;viewport.classList.remove('dragging');try{viewport.releasePointerCapture(e.pointerId)}catch{}}
viewport.addEventListener('pointerdown',e=>{if(!e.altKey&&e.button!==0)return;e.preventDefault();drag={x:e.clientX,y:e.clientY,button:e.button,alt:e.altKey,distance:0};if(e.altKey)viewport.classList.add('dragging');viewport.setPointerCapture(e.pointerId)});
viewport.addEventListener('pointermove',e=>{if(!drag)return;e.preventDefault();const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;drag.distance+=Math.hypot(dx,dy);if(drag.alt)queueMove(drag.button===0?'orbit':drag.button===1?'pan':'zoom',dx,dy)});
viewport.addEventListener('pointerup',endDrag);
viewport.addEventListener('pointercancel',endDrag);
viewport.addEventListener('lostpointercapture',()=>{drag=null;viewport.classList.remove('dragging')});
viewport.addEventListener('dragstart',e=>e.preventDefault());
viewport.addEventListener('contextmenu',e=>e.preventDefault());
viewport.addEventListener('wheel',e=>{e.preventDefault();action({action:'zoom',delta:e.deltaY})},{passive:false});
viewport.addEventListener('dblclick',()=>action({action:'reset'}));
document.querySelectorAll('#modes button').forEach(button=>button.addEventListener('click',async()=>{const mode=button.dataset.mode;try{await fetch('/api/view',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({mode})});document.querySelectorAll('#modes button').forEach(item=>item.classList.toggle('on',item===button))}catch{}}));
async function status(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();document.querySelector('#state').textContent=s.error?'error':s.phase;document.querySelector('#frame').textContent=s.frame;document.querySelector('#physics').textContent=s.physics_frame;document.querySelector('#phase').textContent=s.error?'OVRT render failed':(s.phase==='initializing'?'Starting NVIDIA OVRT':s.phase);document.querySelector('#detail').textContent=s.error||s.detail;if(s.has_image)empty.style.display='none'}catch{document.querySelector('#state').textContent='disconnected'}}
setInterval(status,750);status();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "BlacknodeOVRT/0.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _headers(self, status: int, content_type: str, length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
            self.wfile.write(body)
            return
        if path == "/api/status":
            body = json.dumps(STATE.status()).encode("utf-8")
            self._headers(HTTPStatus.OK, "application/json", len(body))
            self.wfile.write(body)
            return
        if path == "/stream.mjpg":
            self._headers(HTTPStatus.OK, "multipart/x-mixed-replace; boundary=frame")
            seen = -1
            try:
                while not STATE.stop:
                    with STATE.frame_ready:
                        STATE.frame_ready.wait_for(
                            lambda: STATE.frame_number != seen or STATE.stop, timeout=2.0
                        )
                        if STATE.stop:
                            return
                        seen = STATE.frame_number
                        frame = STATE.jpeg
                    if not frame:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            return
        self._headers(HTTPStatus.NOT_FOUND, "text/plain", 0)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/camera", "/api/view", "/api/pick"}:
            self._headers(HTTPStatus.NOT_FOUND, "text/plain", 0)
            return
        try:
            length = min(16384, int(self.headers.get("Content-Length") or 0))
            value = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/pick":
                x = float(value.get("x"))
                y = float(value.get("y"))
                if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                    raise ValueError("pick coordinates must be normalized")
                with STATE.lock:
                    STATE.pick_requests.append({"x": x, "y": y})
                    del STATE.pick_requests[:-8]
                self._headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                return
            if path == "/api/view":
                mode = str(value.get("mode") or "rgb").lower()
                if mode not in {"rgb", "depth", "segmentation", "detection", "composite"}:
                    raise ValueError("unsupported perception view")
                with STATE.lock:
                    STATE.view_mode = mode
                self._headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                return
            action = str(value.get("action") or "")
            if action not in {"orbit", "pan", "zoom", "reset"}:
                raise ValueError("unsupported camera action")
            clean: dict[str, float | str] = {"action": action}
            for name in ("dx", "dy", "delta"):
                if name in value:
                    clean[name] = max(-500.0, min(500.0, float(value[name])))
            with STATE.lock:
                STATE.camera_actions.append(clean)
                del STATE.camera_actions[:-64]
            self._headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
        except (ValueError, TypeError, json.JSONDecodeError):
            self._headers(HTTPStatus.BAD_REQUEST, "text/plain", 0)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def _serve(port: int) -> Server:
    server = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="ovrtx-http").start()
    return server


def _reader() -> dict[str, Any]:
    first = sys.stdin.readline()
    if not first:
        raise RuntimeError("OVRT worker did not receive configuration")
    message = json.loads(first)
    if message.get("type") != "configure":
        raise RuntimeError("OVRT worker expected configure as its first message")
    return dict(message.get("config") or {})


def _start_input_reader() -> None:
    def consume() -> None:
        for raw_line in sys.stdin:
            try:
                event = json.loads(raw_line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "poses":
                with STATE.lock:
                    STATE.latest_transforms = list(event.get("transforms") or [])
                    STATE.physics_frame = int(event.get("frame") or 0)
            elif kind == "reference_pose":
                with STATE.lock:
                    STATE.latest_reference = {
                        "transforms": list(event.get("transforms") or []),
                        "visible": bool(event.get("visible", True)),
                        "offset_m": [
                            float(value)
                            for value in list(event.get("offset_m") or [0.0, 0.0, 0.0])[:3]
                        ],
                    }
            elif kind == "camera":
                action = str(event.get("action") or "")
                if action in {"orbit", "pan", "zoom", "reset"}:
                    clean: dict[str, float | str] = {"action": action}
                    for name in ("dx", "dy", "delta"):
                        if name in event:
                            clean[name] = float(event[name])
                    with STATE.lock:
                        STATE.camera_actions.append(clean)
                        del STATE.camera_actions[:-64]
            elif kind == "view":
                mode = str(event.get("mode") or "rgb").lower()
                if mode in {"rgb", "depth", "segmentation", "detection", "composite"}:
                    with STATE.lock:
                        STATE.view_mode = mode
            elif kind == "visibility":
                path = str(event.get("path") or "").strip()
                if path.startswith("/"):
                    with STATE.lock:
                        STATE.visibility_updates.append({
                            "path": path,
                            "visible": bool(event.get("visible")),
                        })
                        del STATE.visibility_updates[:-256]
            elif kind in {
                "grid", "transform", "material", "environment", "render_options", "gizmo_tool"
            }:
                with STATE.lock:
                    STATE.stage_updates.append(dict(event))
                    del STATE.stage_updates[:-256]
            elif kind == "pick":
                x = float(event.get("x", -1.0))
                y = float(event.get("y", -1.0))
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    with STATE.lock:
                        STATE.pick_requests.append({"x": x, "y": y})
                        del STATE.pick_requests[:-8]
            elif kind == "select":
                path = str(event.get("path") or "")
                with STATE.lock:
                    STATE.selection_requests.append(path)
                    del STATE.selection_requests[:-8]
            elif kind == "close":
                with STATE.frame_ready:
                    STATE.stop = True
                    STATE.frame_ready.notify_all()
                return
        with STATE.frame_ready:
            STATE.stop = True
            STATE.frame_ready.notify_all()

    threading.Thread(target=consume, daemon=True, name="ovrtx-input").start()


def _hex_color(value: str) -> tuple[float, float, float]:
    clean = str(value).strip().lstrip("#")
    if len(clean) != 6 or any(char not in "0123456789abcdefABCDEF" for char in clean):
        clean = "6383c5"
    return tuple(int(clean[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def _asset_reference(path: str) -> str:
    """Serialize an absolute path using OpenUSD's asset-path delimiters."""
    identifier = str(Path(path).resolve()).replace("\\", "/")
    if "@" in identifier:
        return "@@@" + identifier.replace("@@@", r"\@@@") + "@@@"
    return f"@{identifier}@"


def _matrix_text(values: list[float]) -> str:
    rows = [values[offset:offset + 4] for offset in range(0, 16, 4)]
    return "(" + ", ".join("(" + ", ".join(f"{value:.12g}" for value in row) + ")" for row in rows) + ")"


def _camera_matrix(eye: list[float], target: list[float], up_axis: str) -> list[float]:
    import numpy as np

    eye_array = np.asarray(eye, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    forward = target_array - eye_array
    length = float(np.linalg.norm(forward))
    if length < 1.0e-6:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward /= length
    up = np.zeros(3, dtype=np.float64)
    up[{"x": 0, "y": 1, "z": 2}.get(up_axis, 2)] = 1.0
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1.0e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return [
        *right.tolist(), 0.0,
        *true_up.tolist(), 0.0,
        *(-forward).tolist(), 0.0,
        *eye_array.tolist(), 1.0,
    ]


def _grid_usda(height: float, extent: float, up_axis: str, meters_per_unit: float) -> str:
    points: list[str] = []
    counts: list[str] = []
    half_extent = max(0.01, float(extent))
    spacing = half_extent / 10.0
    for value in range(-10, 11):
        coordinate = value * spacing
        if up_axis == "y":
            points.extend([
                f"({-half_extent:.6g}, {height:.6g}, {coordinate:.6g})",
                f"({half_extent:.6g}, {height:.6g}, {coordinate:.6g})",
                f"({coordinate:.6g}, {height:.6g}, {-half_extent:.6g})",
                f"({coordinate:.6g}, {height:.6g}, {half_extent:.6g})",
            ])
        else:
            points.extend([
                f"({-half_extent:.6g}, {coordinate:.6g}, {height:.6g})",
                f"({half_extent:.6g}, {coordinate:.6g}, {height:.6g})",
                f"({coordinate:.6g}, {-half_extent:.6g}, {height:.6g})",
                f"({coordinate:.6g}, {half_extent:.6g}, {height:.6g})",
            ])
        counts.extend(["2", "2"])
    lift = 0.0005 / max(1.0e-12, meters_per_unit)
    width = 0.0015 / max(1.0e-12, meters_per_unit)
    if up_axis == "y":
        axis_points = (
            f"({-half_extent:.6g}, {height + lift:.6g}, 0), ({half_extent:.6g}, {height + lift:.6g}, 0), "
            f"(0, {height + lift:.6g}, {-half_extent:.6g}), (0, {height + lift:.6g}, {half_extent:.6g})"
        )
    else:
        axis_points = (
            f"({-half_extent:.6g}, 0, {height + lift:.6g}), ({half_extent:.6g}, 0, {height + lift:.6g}), "
            f"(0, {-half_extent:.6g}, {height + lift:.6g}), (0, {half_extent:.6g}, {height + lift:.6g})"
        )
    return f'''\n    def BasisCurves "Grid"\n    {{\n        uniform token type = "linear"\n        uniform token wrap = "nonperiodic"\n        int[] curveVertexCounts = [{", ".join(counts)}]\n        point3f[] points = [{", ".join(points)}]\n        float[] widths = [{width:.6g}] (\n            interpolation = "constant"\n        )\n        color3f[] primvars:displayColor = [(0.24, 0.28, 0.34)]\n    }}\n\n    def BasisCurves "GridAxes"\n    {{\n        uniform token type = "linear"\n        uniform token wrap = "nonperiodic"\n        int[] curveVertexCounts = [2, 2]\n        point3f[] points = [{axis_points}]\n        float[] widths = [{width * 2.0:.6g}] (\n            interpolation = "constant"\n        )\n        color3f[] primvars:displayColor = [(0.72, 0.16, 0.12), (0.12, 0.28, 0.72)] (\n            interpolation = "uniform"\n        )\n    }}\n'''


def _ghost_usda(config: dict[str, Any]) -> str:
    """Author translucent reference instances for the controlled articulation."""
    entries = list(config.get("ghost_entries") or [])
    source = str(config.get("asset_path") or "")
    if not entries or not source:
        return ""
    instances: list[str] = []
    for entry in entries:
        name = Path(str(entry.get("path") or "")).name
        source_path = str(entry.get("source_path") or "")
        if not name or not source_path.startswith("/"):
            continue
        instances.append(f'''
        over "{name}" (
            prepend references = {_asset_reference(source)}<{source_path}>
        )
        {{
            token visibility = "invisible"
            rel material:binding = </BlacknodeOVRT/RealGhostMaterial> (
                bindMaterialAs = "strongerThanDescendants"
            )
            matrix4d xformOp:transform = {_matrix_text([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
''')
    if not instances:
        return ""
    return f'''
    def Material "RealGhostMaterial"
    {{
        token outputs:surface.connect = </BlacknodeOVRT/RealGhostMaterial/PreviewSurface.outputs:surface>

        def Shader "PreviewSurface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.18, 0.86, 1)
            float inputs:metallic = 0
            float inputs:roughness = 0.35
            float inputs:opacity = 0.28
            token outputs:surface
        }}
    }}

    def Scope "RealGhost"
    {{{''.join(instances)}
    }}
'''


def _gizmo_usda() -> str:
    """Author compact native OVRT gizmos; browser controls remain hit targets only."""
    ring_counts: list[str] = []
    ring_points: list[str] = []
    samples = 64
    for axis in range(3):
        ring_counts.append(str(samples + 1))
        for sample in range(samples + 1):
            angle = math.tau * sample / samples
            cosine = math.cos(angle) * 0.82
            sine = math.sin(angle) * 0.82
            point = (
                (0.0, cosine, sine)
                if axis == 0
                else (cosine, 0.0, sine)
                if axis == 1
                else (cosine, sine, 0.0)
            )
            ring_points.append(f"({point[0]:.6g}, {point[1]:.6g}, {point[2]:.6g})")
    colors = "[(1, 0.16, 0.12), (0.2, 0.86, 0.22), (0.12, 0.42, 1)]"
    axis_points = "[(0,0,0),(.82,0,0), (0,0,0),(0,.82,0), (0,0,0),(0,0,.82)]"
    def tip(kind: str, name: str, axis: str, position: tuple[float, float, float], color: str) -> str:
        size = "double height = 0.22\n            double radius = 0.075" if kind == "Cone" else "double size = 0.14"
        axis_value = f'uniform token axis = "{axis}"\n            ' if kind == "Cone" else ""
        return f'''        def {kind} "{name}"
        {{
            {axis_value}{size}
            color3f[] primvars:displayColor = [{color}]
            double3 xformOp:translate = ({position[0]:.6g}, {position[1]:.6g}, {position[2]:.6g})
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
'''
    move_tips = "".join((
        tip("Cone", "XTip", "X", (0.93, 0.0, 0.0), "(1, 0.16, 0.12)"),
        tip("Cone", "YTip", "Y", (0.0, 0.93, 0.0), "(0.2, 0.86, 0.22)"),
        tip("Cone", "ZTip", "Z", (0.0, 0.0, 0.93), "(0.12, 0.42, 1)"),
    ))
    scale_tips = "".join((
        tip("Cube", "XHandle", "X", (0.94, 0.0, 0.0), "(1, 0.16, 0.12)"),
        tip("Cube", "YHandle", "Y", (0.0, 0.94, 0.0), "(0.2, 0.86, 0.22)"),
        tip("Cube", "ZHandle", "Z", (0.0, 0.0, 0.94), "(0.12, 0.42, 1)"),
        tip("Cube", "Uniform", "X", (0.0, 0.0, 0.0), "(1, 0.68, 0.08)"),
    ))
    axes = f"""
        def BasisCurves "Axes"
        {{
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [2, 2, 2]
            point3f[] points = {axis_points}
            float[] widths = [0.025, 0.025, 0.025] (interpolation = "uniform")
            color3f[] primvars:displayColor = {colors} (interpolation = "uniform")
        }}
    """
    identity = _matrix_text([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    return f"""
    def Xform "GizmoMove"
    {{
        token visibility = "invisible"
        matrix4d xformOp:transform = {identity}
        uniform token[] xformOpOrder = ["xformOp:transform"]
{axes}{move_tips}
    }}
    def Xform "GizmoRotate"
    {{
        token visibility = "invisible"
        matrix4d xformOp:transform = {identity}
        uniform token[] xformOpOrder = ["xformOp:transform"]
        def BasisCurves "Rings"
        {{
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [{", ".join(ring_counts)}]
            point3f[] points = [{", ".join(ring_points)}]
            float[] widths = [0.025, 0.025, 0.025] (interpolation = "uniform")
            color3f[] primvars:displayColor = {colors} (interpolation = "uniform")
        }}
    }}
    def Xform "GizmoScale"
    {{
        token visibility = "invisible"
        matrix4d xformOp:transform = {identity}
        uniform token[] xformOpOrder = ["xformOp:transform"]
{axes}{scale_tips}
    }}
"""


def _build_wrapper_usda(config: dict[str, Any], camera_matrix: list[float]) -> str:
    source = str(config.get("asset_path") or "")
    sublayers = f"subLayers = [{_asset_reference(source)}]" if source else ""
    background = _hex_color(str(config.get("background_color") or "#6383c5"))
    width = int(config["width"])
    height = int(config["height"])
    meters_per_unit = max(1.0e-12, float(config.get("meters_per_unit") or 1.0))
    up_axis = "y" if str(config.get("up_axis") or "z").lower() == "y" else "z"
    ground_height = float(config.get("ground_height") or 0.0) / meters_per_unit
    body_defs: list[str] = []
    for entry in config.get("body_entries") or []:
        if str(entry.get("path") or "").startswith("/BlacknodeOVRT/RigidBodies/"):
            name = Path(str(entry["path"])).name
            color = [float(value) for value in entry.get("color", [0.8, 0.25, 0.12])]
            label = str(entry.get("name") or name)
            body_defs.append(f'''\n        def Cube "{name}" (\n            prepend apiSchemas = ["SemanticsAPI:class", "SemanticsAPI:label"]\n        )\n        {{\n            double size = 1\n            string semantic:class:params:semanticData = "rigid_body"\n            string semantic:class:params:semanticType = "class"\n            string semantic:label:params:semanticData = "{label}"\n            string semantic:label:params:semanticType = "label"\n            color3f[] primvars:displayColor = [({color[0]:.6g}, {color[1]:.6g}, {color[2]:.6g})]\n            matrix4d xformOp:transform = {_matrix_text([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])}\n            uniform token[] xformOpOrder = ["xformOp:transform"]\n        }}\n''')
    grid = _grid_usda(
        ground_height,
        float(config.get("grid_extent") or 1.0 / meters_per_unit),
        up_axis,
        meters_per_unit,
    )
    grid += _ghost_usda(config)
    grid += _gizmo_usda()
    ground = ""
    if bool(config.get("ground_enabled", True)):
        horizontal = max(
            2.0 / meters_per_unit,
            float(config.get("grid_extent") or 1.0 / meters_per_unit) * 2.0,
        )
        thickness = 0.02 / meters_per_unit
        default_translation = (
            [0.0, ground_height * meters_per_unit, 0.0]
            if up_axis == "y"
            else [0.0, 0.0, ground_height * meters_per_unit]
        )
        ground_transform = dict(config.get("ground_transform") or {
            "translate_m": default_translation,
            "rotate_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        })
        ground_matrix = _editor_matrix(ground_transform, meters_per_unit)
        if up_axis == "y":
            geometry_matrix = [horizontal,0,0,0, 0,thickness,0,0, 0,0,horizontal,0, 0,-thickness / 2.0,0,1]
        else:
            geometry_matrix = [horizontal,0,0,0, 0,horizontal,0,0, 0,0,thickness,0, 0,0,-thickness / 2.0,1]
        material = {
            "base_color": [0.18, 0.2, 0.24],
            "metallic": 0.0,
            "roughness": 0.8,
            "opacity": 1.0,
            **dict(config.get("ground_material") or {}),
        }
        color = [float(value) for value in material["base_color"]]
        ground = f'''\n    def Material "GroundMaterial"\n    {{\n        token outputs:surface.connect = </BlacknodeOVRT/GroundMaterial/PreviewSurface.outputs:surface>\n\n        def Shader "PreviewSurface"\n        {{\n            uniform token info:id = "UsdPreviewSurface"\n            color3f inputs:diffuseColor = ({color[0]:.6g}, {color[1]:.6g}, {color[2]:.6g})\n            float inputs:metallic = {float(material["metallic"]):.6g}\n            float inputs:roughness = {float(material["roughness"]):.6g}\n            float inputs:opacity = {float(material["opacity"]):.6g}\n            token outputs:surface\n        }}\n    }}\n\n    def Xform "Ground" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {{\n        rel material:binding = </BlacknodeOVRT/GroundMaterial>\n        matrix4d xformOp:transform = {_matrix_text(ground_matrix)}\n        uniform token[] xformOpOrder = ["xformOp:transform"]\n\n        def Cube "Geometry"\n        {{\n            double size = 1\n            color3f[] primvars:displayColor = [({color[0]:.6g}, {color[1]:.6g}, {color[2]:.6g})]\n            matrix4d xformOp:transform = {_matrix_text(geometry_matrix)}\n            uniform token[] xformOpOrder = ["xformOp:transform"]\n        }}\n    }}\n'''
    return f'''#usda 1.0\n(\n    {sublayers}\n    upAxis = "Z"\n    metersPerUnit = 1\n)\n\ndef Xform "BlacknodeOVRT"\n{{\n    def Camera "Camera" (\n        prepend apiSchemas = ["OmniSensorGenericCameraCoreAPI"]\n    )\n    {{\n        float focalLength = 32\n        float horizontalAperture = 36\n        float2 clippingRange = (0.01, 1000)\n        matrix4d xformOp:transform = {_matrix_text(camera_matrix)}\n        uniform token[] xformOpOrder = ["xformOp:transform"]\n    }}\n\n    def DistantLight "Key"\n    {{\n        float intensity = 2500\n        float angle = 4\n        color3f color = (1, 0.95, 0.88)\n        float xformOp:rotateXYZ = (-35, 25, -25)\n        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n    }}\n\n    def DomeLight "Sky"\n    {{\n        float intensity = 450\n        color3f color = ({background[0]:.6g}, {background[1]:.6g}, {background[2]:.6g})\n    }}\n{ground}{grid}\n    def Scope "RigidBodies"\n    {{{''.join(body_defs)}\n    }}\n\n    def Scope "Render"\n    {{\n        def RenderProduct "Viewport"\n        {{\n            int2 resolution = ({width}, {height})\n            rel camera = </BlacknodeOVRT/Camera>\n            rel orderedVars = [<LdrColor>, <HdrColor>]\n\n            def RenderVar "LdrColor"\n            {{\n                string sourceName = "LdrColor"\n            }}\n\n            def RenderVar "HdrColor"\n            {{\n                string sourceName = "HdrColor"\n            }}\n        }}\n    }}\n}}\n'''


def _wrapper_usda(config: dict[str, Any], camera_matrix: list[float]) -> str:
    """Build one composed root layer while preserving source stage semantics."""
    text = _build_wrapper_usda(config, camera_matrix)
    text = (
        text.replace("float intensity = 2500", "float inputs:intensity = 2500")
        .replace("color3f color = (1, 0.95, 0.88)", "color3f inputs:color = (1, 0.95, 0.88)")
        .replace("float angle = 4", "float inputs:angle = 4")
        .replace("float intensity = 450", "float inputs:intensity = 450")
        .replace("color3f color = (", "color3f inputs:color = (")
        .replace(
            "def RenderProduct \"Viewport\"\n        {\n",
            "def RenderProduct \"Viewport\"\n        {\n            uint[] deviceIds = [0]\n",
        )
    )
    text = text.replace(
        '        def RenderProduct "Viewport"\n        {\n            uint[] deviceIds = [0]\n',
        '        def RenderProduct "Viewport" (\n'
        '            prepend apiSchemas = ["OmniRtxSettingsCommonAPI_1"]\n'
        '        )\n        {\n            uint[] deviceIds = [0]\n',
    )
    text = text.replace(
        "rel orderedVars = [<LdrColor>, <HdrColor>]",
        "rel orderedVars = [<LdrColor>, <HdrColor>, <DistanceToCameraSD>, "
        "<SemanticSegmentation>, <SemanticIdMap>]",
    )
    render_var_marker = '''            def RenderVar "HdrColor"
            {
                string sourceName = "HdrColor"
            }
'''
    perception_render_vars = '''
            def RenderVar "DistanceToCameraSD"
            {
                string sourceName = "DistanceToCameraSD"
            }

            def RenderVar "SemanticSegmentation"
            {
                string sourceName = "SemanticSegmentation"
            }

            def RenderVar "SemanticIdMap"
            {
                string sourceName = "SemanticIdMap"
            }
'''
    text = text.replace(render_var_marker, render_var_marker + perception_render_vars)
    environment = dict(config.get("environment") or {})
    distant = _distant_light(environment)
    distant_intensity = distant["intensity"] if distant["enabled"] else 0.0
    distant_color = distant["color"]
    distant_rotation = distant["rotation_deg"]
    text = (
        text.replace(
            "float inputs:intensity = 2500",
            f"float inputs:intensity = {distant_intensity:.6g}",
        )
        .replace(
            "color3f inputs:color = (1, 0.95, 0.88)",
            "color3f inputs:color = "
            f"({distant_color[0]:.6g}, {distant_color[1]:.6g}, {distant_color[2]:.6g})",
        )
        .replace("float inputs:angle = 4", f"float inputs:angle = {distant['angle_deg']:.6g}")
        .replace(
            "float xformOp:rotateXYZ = (-35, 25, -25)",
            "float xformOp:rotateXYZ = "
            f"({distant_rotation[0]:.6g}, {distant_rotation[1]:.6g}, {distant_rotation[2]:.6g})",
        )
    )
    hdri_path = _environment_texture_path(environment)
    background_source = _background_source_type(environment)
    background = _hex_color(str(environment.get("background_color") or config.get("background_color") or "#6383c5"))
    text = text.replace(
        "            uint[] deviceIds = [0]\n            int2 resolution",
        "            uint[] deviceIds = [0]\n"
        f"            int omni:rtx:background:source:type = {background_source}\n"
        + (
            f"            asset omni:rtx:background:source:texture:path = {_asset_reference(hdri_path)}\n"
            if hdri_path and Path(hdri_path).is_file()
            else ""
        )
        + f"            color3f omni:rtx:background:source:color = ({background[0]:.6g}, {background[1]:.6g}, {background[2]:.6g})\n"
        + "            int2 resolution",
    )
    # The base dome supplies the solid-color fallback. HDRIs are authored as a
    # replaceable population reference because scalar asset replacement is not
    # reliably represented by a generic live attribute write in OVStage 0.1.
    # Keep the distant key/sun enabled independently.
    text = text.replace(
        "float inputs:intensity = 450",
        f"float inputs:intensity = "
        f"{(0.0 if hdri_path else _dome_light_intensity(environment)):.6g}",
    )
    text = text.replace(
        '    def DomeLight "Sky"\n    {\n',
        '    def DomeLight "Sky"\n    {\n'
        + "        bool inputs:visibleInPrimaryRay = "
        + ("true" if bool(environment.get("show_background", True)) else "false")
        + "\n",
    )
    if not bool(config.get("show_grid", True)):
        for name in ("Grid", "GridAxes"):
            text = text.replace(
                f'    def BasisCurves "{name}"\n    {{\n',
                f'    def BasisCurves "{name}"\n    {{\n        token visibility = "invisible"\n',
            )
    meters_per_unit = max(1.0e-12, float(config.get("meters_per_unit") or 1.0))
    up_axis = "Y" if str(config.get("up_axis") or "z").lower() == "y" else "Z"
    clipping_range = f"({0.01 / meters_per_unit:.12g}, {1000.0 / meters_per_unit:.12g})"
    return (
        text.replace('upAxis = "Z"', f'upAxis = "{up_axis}"')
        .replace("metersPerUnit = 1", f"metersPerUnit = {meters_per_unit:.12g}")
        .replace("float2 clippingRange = (0.01, 1000)", f"float2 clippingRange = {clipping_range}")
        .replace("float xformOp:rotateXYZ", "float3 xformOp:rotateXYZ")
    )


def _pose_matrix(
    transform: list[float],
    scale: list[float],
    meters_per_unit: float = 1.0,
    scale_in_meters: bool = False,
) -> list[float]:
    x, y, z, qx, qy, qz, qw = (float(value) for value in transform[:7])
    sx, sy, sz = (float(value) for value in scale)
    units = max(1.0e-12, float(meters_per_unit))
    x, y, z = x / units, y / units, z / units
    if scale_in_meters:
        sx, sy, sz = sx / units, sy / units, sz / units
    # Newton/Warp stores quaternions as x, y, z, w. USD matrices use row vectors.
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        sx * (1 - 2 * (yy + zz)), sx * 2 * (xy + wz), sx * 2 * (xz - wy), 0,
        sy * 2 * (xy - wz), sy * (1 - 2 * (xx + zz)), sy * 2 * (yz + wx), 0,
        sz * 2 * (xz + wy), sz * 2 * (yz - wx), sz * (1 - 2 * (xx + yy)), 0,
        x, y, z, 1,
    ]


def _body_pose_matrix(
    entry: dict[str, Any], transforms: list[list[float]], meters_per_unit: float
) -> list[float]:
    """Convert Newton world body poses to the local USD xform authored at the body prim."""
    import numpy as np

    if (
        len(list(entry.get("initial_world_matrix") or [])) == 16
        and len(list(entry.get("body_bind_pose") or [])) >= 7
    ):
        bind_entry = dict(entry)
        bind_entry["body_index"] = int(entry["index"])
        bind_entry["body_bind_world_matrix"] = _pose_matrix(
            list(entry["body_bind_pose"]), [1.0, 1.0, 1.0], meters_per_unit
        )
        return _bound_shape_pose_matrix(bind_entry, transforms, meters_per_unit)

    index = int(entry["index"])
    child_world = np.asarray(
        _pose_matrix(
            transforms[index],
            list(entry.get("world_scale") or entry["scale"]),
            meters_per_unit,
            bool(entry.get("scale_in_meters", False)),
        ),
        dtype=np.float64,
    ).reshape(4, 4)
    local_pose = list(entry.get("local_pose") or [])
    if len(local_pose) == 7:
        shape_local = np.asarray(
            _pose_matrix(local_pose, [1.0, 1.0, 1.0], meters_per_unit),
            dtype=np.float64,
        ).reshape(4, 4)
        child_world = shape_local @ child_world
    render_parent_relative = list(entry.get("render_parent_relative") or [])
    if len(render_parent_relative) == 16:
        parent_world = np.asarray(
            render_parent_relative, dtype=np.float64
        ).reshape(4, 4)
        render_parent_index = int(entry.get("render_parent_index", -1))
        if 0 <= render_parent_index < len(transforms):
            parent_world = parent_world @ np.asarray(
                _pose_matrix(
                    transforms[render_parent_index],
                    list(entry.get("render_parent_world_scale") or [1.0, 1.0, 1.0]),
                    meters_per_unit,
                ),
                dtype=np.float64,
            ).reshape(4, 4)
        local = child_world @ np.linalg.inv(parent_world)
        return local.reshape(-1).tolist()
    parent_index = int(entry.get("parent_index", -1))
    if parent_index < 0 or parent_index >= len(transforms):
        return child_world.reshape(-1).tolist()
    parent_world = np.asarray(
        _pose_matrix(transforms[parent_index], [1.0, 1.0, 1.0], meters_per_unit),
        dtype=np.float64,
    ).reshape(4, 4)
    # USD composes row-vector transforms as local * parent-to-world.
    local = child_world @ np.linalg.inv(parent_world)
    return local.reshape(-1).tolist()


def _bound_shape_pose_matrix(
    entry: dict[str, Any],
    transforms: list[list[float]],
    meters_per_unit: float,
    bind_world_delta: list[float] | None = None,
) -> list[float]:
    """Place one USD Gprim from its authored body frame at Newton's live pose."""
    import numpy as np

    body_index = int(entry["body_index"])
    shape_bind_world = np.asarray(
        entry["initial_world_matrix"], dtype=np.float64
    ).reshape(4, 4)
    if bind_world_delta is not None and len(bind_world_delta) == 16:
        shape_bind_world = shape_bind_world @ np.asarray(
            bind_world_delta, dtype=np.float64
        ).reshape(4, 4)
    body_bind_world = np.asarray(
        entry["body_bind_world_matrix"], dtype=np.float64
    ).reshape(4, 4)
    body_current = np.asarray(
        _pose_matrix(transforms[body_index], [1.0, 1.0, 1.0], meters_per_unit),
        dtype=np.float64,
    ).reshape(4, 4)
    shape_body_relative = shape_bind_world @ np.linalg.inv(body_bind_world)
    desired_world = shape_body_relative @ body_current
    parent_values = list(entry.get("render_parent_world_matrix") or [])
    if len(parent_values) == 16:
        parent_world = np.asarray(parent_values, dtype=np.float64).reshape(4, 4)
        # The target is a nested source Gprim. OVStage consumes its authored
        # parent-relative matrix even though container writes themselves do not
        # propagate reliably through the populated hierarchy.
        desired_world = desired_world @ np.linalg.inv(parent_world)
    return desired_world.reshape(-1).tolist()


def _interaction_transform_delta(
    path: str,
    frames: dict[str, dict[str, Any]],
    transform_overrides: dict[str, list[float]],
) -> list[float] | None:
    """Resolve the nearest edited USD ancestor as a bind-world delta."""
    import numpy as np

    candidate = str(path or "").rstrip("/")
    while candidate:
        override = transform_overrides.get(candidate)
        frame = frames.get(candidate)
        if override is not None and frame is not None:
            local = np.asarray(frame.get("local_matrix") or [], dtype=np.float64)
            parent = np.asarray(frame.get("parent_world") or [], dtype=np.float64)
            changed = np.asarray(override, dtype=np.float64)
            if local.size == parent.size == changed.size == 16:
                old_world = local.reshape(4, 4) @ parent.reshape(4, 4)
                new_world = changed.reshape(4, 4) @ parent.reshape(4, 4)
                return (np.linalg.inv(old_world) @ new_world).reshape(-1).tolist()
        candidate = candidate.rsplit("/", 1)[0]
    return None


class Camera:
    def __init__(self, config: dict[str, Any]) -> None:
        import numpy as np

        camera = dict(config.get("camera") or {})
        self.eye = np.asarray(camera.get("position", [1.0, -1.0, 0.8]), dtype=np.float64)
        self.target = np.asarray(camera.get("target", [0.0, 0.0, 0.2]), dtype=np.float64)
        self.initial_eye = self.eye.copy()
        self.initial_target = self.target.copy()
        self.up_axis = str(camera.get("up_axis") or "z")

    def matrix(self) -> list[float]:
        return _camera_matrix(self.eye.tolist(), self.target.tolist(), self.up_axis)

    def project(self, point: Any, width: int, height: int) -> list[float] | None:
        """Project a stage-space point to top-left-origin normalized viewport coordinates."""
        import numpy as np

        offset = np.asarray(point, dtype=np.float64) - self.eye
        forward = self.target - self.eye
        distance = float(np.linalg.norm(forward))
        if distance < 1.0e-9:
            return None
        forward /= distance
        world_up = np.zeros(3, dtype=np.float64)
        world_up[{"x": 0, "y": 1, "z": 2}.get(self.up_axis, 2)] = 1.0
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1.0e-9:
            return None
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        depth = float(np.dot(offset, forward))
        if depth <= 1.0e-6:
            return None
        tan_half_horizontal = 36.0 / (2.0 * 32.0)
        tan_half_vertical = tan_half_horizontal * max(1, int(height)) / max(1, int(width))
        return [
            0.5 + float(np.dot(offset, right)) / (2.0 * depth * tan_half_horizontal),
            0.5 - float(np.dot(offset, up)) / (2.0 * depth * tan_half_vertical),
        ]

    def apply(self, actions: list[dict[str, float | str]]) -> bool:
        import numpy as np

        dirty = False
        for action in actions:
            kind = action.get("action")
            if kind == "reset":
                self.eye = self.initial_eye.copy()
                self.target = self.initial_target.copy()
                dirty = True
                continue
            offset = self.eye - self.target
            distance = max(0.05, float(np.linalg.norm(offset)))
            forward = -offset / distance
            world_up = np.zeros(3)
            world_up[{"x": 0, "y": 1, "z": 2}.get(self.up_axis, 2)] = 1.0
            right = np.cross(forward, world_up)
            if float(np.linalg.norm(right)) < 1.0e-6:
                continue
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            if kind == "zoom":
                factor = math.exp(float(action.get("delta", 0.0)) * 0.0015)
                self.eye = self.target + offset * factor
                dirty = True
            elif kind == "pan":
                motion = (-right * float(action.get("dx", 0.0)) + up * float(action.get("dy", 0.0))) * distance * 0.0012
                self.eye += motion
                self.target += motion
                dirty = True
            elif kind == "orbit":
                yaw = -float(action.get("dx", 0.0)) * 0.006
                pitch = -float(action.get("dy", 0.0)) * 0.006
                def rotate(vector: Any, axis: Any, angle: float) -> Any:
                    axis = axis / np.linalg.norm(axis)
                    return vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle) + axis * np.dot(axis, vector) * (1 - math.cos(angle))
                offset = rotate(offset, world_up, yaw)
                offset = rotate(offset, right, pitch)
                self.eye = self.target + offset
                dirty = True
        return dirty

    def environment_background(
        self, image: Any, width: int, height: int, intensity: float
    ) -> Any:
        """Project a lat-long HDRI into the current pinhole camera."""
        import cv2
        import numpy as np

        forward = self.target - self.eye
        forward /= max(1.0e-12, float(np.linalg.norm(forward)))
        world_up = np.zeros(3, dtype=np.float64)
        up_index = {"x": 0, "y": 1, "z": 2}.get(self.up_axis, 2)
        world_up[up_index] = 1.0
        right = np.cross(forward, world_up)
        right /= max(1.0e-12, float(np.linalg.norm(right)))
        up = np.cross(right, forward)
        aspect = float(width) / max(1.0, float(height))
        tan_half_hfov = math.tan(math.atan(36.0 / (2.0 * 32.0)))
        xs = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
        ys = 1.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0
        grid_x, grid_y = np.meshgrid(xs, ys)
        rays = (
            forward[None, None, :]
            + grid_x[..., None] * tan_half_hfov * right[None, None, :]
            + grid_y[..., None] * (tan_half_hfov / aspect) * up[None, None, :]
        )
        rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1.0e-12)
        if self.up_axis == "y":
            longitude = np.arctan2(rays[..., 2], rays[..., 0])
            latitude = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
        else:
            longitude = np.arctan2(rays[..., 1], rays[..., 0])
            latitude = np.arcsin(np.clip(rays[..., 2], -1.0, 1.0))
        map_x = ((longitude / (2.0 * math.pi) + 0.5) % 1.0) * float(image.shape[1] - 1)
        map_y = (0.5 - latitude / math.pi) * float(image.shape[0] - 1)
        sampled = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        linear = np.maximum(sampled * max(0.0, float(intensity)), 0.0)
        mapped = linear / (1.0 + linear)
        return np.clip(np.power(mapped, 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)


def _resolve_interaction_frame(
    path: str, frames: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    candidate = str(path or "").rstrip("/")
    while candidate:
        frame = frames.get(candidate)
        if frame is not None:
            return frame
        candidate = candidate.rsplit("/", 1)[0]
    return None


def _selection_gizmo(
    camera: Camera,
    frame: dict[str, Any] | None,
    config: dict[str, Any],
    transform_overrides: dict[str, list[float]],
    fallback: list[float] | None = None,
    body_transforms: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Project world axes and object-space rotation rings for a Maya-style gizmo."""
    import numpy as np

    width = max(1, int(config.get("width") or 1))
    height = max(1, int(config.get("height") or 1))
    meters_per_unit = max(1.0e-12, float(config.get("meters_per_unit") or 1.0))
    if frame is None:
        return {"x": float((fallback or [0.5, 0.5])[0]), "y": float((fallback or [0.5, 0.5])[1]), "axes": {}}
    parent_world = np.asarray(frame.get("parent_world") or [], dtype=np.float64)
    local_matrix = np.asarray(
        transform_overrides.get(str(frame.get("path") or ""))
        or frame.get("local_matrix")
        or [],
        dtype=np.float64,
    )
    body_index = int(frame.get("physics_body_index", -1))
    if body_transforms is not None and 0 <= body_index < len(body_transforms):
        parent_world = np.eye(4, dtype=np.float64)
        local_matrix = np.asarray(
            _pose_matrix(
                body_transforms[body_index],
                [1.0, 1.0, 1.0],
                meters_per_unit,
            ),
            dtype=np.float64,
        )
    if parent_world.size != 16 or local_matrix.size != 16:
        return {"x": float((fallback or [0.5, 0.5])[0]), "y": float((fallback or [0.5, 0.5])[1]), "axes": {}}
    parent_world = parent_world.reshape(4, 4)
    local_matrix = local_matrix.reshape(4, 4)
    pivot = (np.asarray([0.0, 0.0, 0.0, 1.0]) @ local_matrix @ parent_world)[:3]
    center = camera.project(pivot, width, height)
    if center is None:
        return {}
    distance_m = max(0.05, float(np.linalg.norm(pivot - camera.eye)) * meters_per_unit)
    sample_m = max(0.01, min(1.0, distance_m * 0.12))
    axes: dict[str, Any] = {}
    local_axes: dict[str, Any] = {}
    for axis, name in enumerate("xyz"):
        local_delta = np.zeros(4, dtype=np.float64)
        local_delta[axis] = sample_m / meters_per_unit
        world_delta = (local_delta @ parent_world)[:3]
        endpoint = camera.project(pivot + world_delta, width, height)
        if endpoint is None:
            continue
        screen = np.asarray([
            (endpoint[0] - center[0]) * width,
            (endpoint[1] - center[1]) * height,
        ])
        pixel_length = float(np.linalg.norm(screen))
        if pixel_length < 1.0e-3:
            continue
        axes[name] = {
            "screen": (screen / pixel_length).tolist(),
            "pixels_per_meter": pixel_length / sample_m,
        }
        local_direction = np.zeros(4, dtype=np.float64)
        local_direction[axis] = 1.0
        world_direction = (local_direction @ local_matrix @ parent_world)[:3]
        direction_length = float(np.linalg.norm(world_direction))
        if direction_length < 1.0e-12:
            continue
        local_endpoint = camera.project(
            pivot + world_direction / direction_length * (sample_m / meters_per_unit),
            width,
            height,
        )
        if local_endpoint is None:
            continue
        local_screen = np.asarray([
            (local_endpoint[0] - center[0]) * width,
            (local_endpoint[1] - center[1]) * height,
        ])
        local_pixel_length = float(np.linalg.norm(local_screen))
        if local_pixel_length < 1.0e-3:
            continue
        local_axes[name] = {
            "screen": (local_screen / local_pixel_length).tolist(),
            "pixels_per_meter": local_pixel_length / sample_m,
        }
        normal = world_direction / direction_length
        seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(normal, seed))) > 0.9:
            seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        ring_u = np.cross(normal, seed)
        ring_u /= max(1.0e-12, float(np.linalg.norm(ring_u)))
        ring_v = np.cross(normal, ring_u)
        ring_points: list[list[float]] = []
        for angle in np.linspace(0.0, math.pi * 2.0, 33):
            ring_point = pivot + (
                ring_u * math.cos(float(angle)) + ring_v * math.sin(float(angle))
            ) * (sample_m / meters_per_unit)
            projected = camera.project(ring_point, width, height)
            if projected is None:
                ring_points = []
                break
            ring_points.append([
                (projected[0] - center[0]) * width,
                (projected[1] - center[1]) * height,
            ])
        radius = max((float(np.linalg.norm(point)) for point in ring_points), default=0.0)
        if radius > 1.0e-3:
            local_axes[name]["ring"] = [
                [point[0] * 44.0 / radius, point[1] * 44.0 / radius]
                for point in ring_points
            ]
    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "axes": axes,
        "local_axes": local_axes,
    }


def _native_gizmo_matrices(
    camera: Camera,
    frame: dict[str, Any] | None,
    config: dict[str, Any],
    transform_overrides: dict[str, list[float]],
    body_transforms: list[list[float]] | None = None,
) -> list[list[float]]:
    """Place native move/world and rotate-scale/local gizmos at constant screen size."""
    import numpy as np

    if frame is None:
        return []
    meters_per_unit = max(1.0e-12, float(config.get("meters_per_unit") or 1.0))
    parent_world = np.asarray(frame.get("parent_world") or [], dtype=np.float64)
    local_matrix = np.asarray(
        transform_overrides.get(str(frame.get("path") or ""))
        or frame.get("local_matrix")
        or [],
        dtype=np.float64,
    )
    body_index = int(frame.get("physics_body_index", -1))
    if body_transforms is not None and 0 <= body_index < len(body_transforms):
        parent_world = np.eye(4, dtype=np.float64)
        local_matrix = np.asarray(
            _pose_matrix(body_transforms[body_index], [1.0, 1.0, 1.0], meters_per_unit),
            dtype=np.float64,
        )
    if parent_world.size != 16 or local_matrix.size != 16:
        return []
    world = local_matrix.reshape(4, 4) @ parent_world.reshape(4, 4)
    pivot = world[3, :3]
    distance_m = max(0.05, float(np.linalg.norm(pivot - camera.eye)) * meters_per_unit)
    # The 32 mm / 36 mm camera projects 0.034 * distance to roughly 58 px at
    # 1080p, matching the invisible browser hit shafts.
    scale = max(0.005, min(0.5, distance_m * 0.034)) / meters_per_unit
    move = np.eye(4, dtype=np.float64)
    move[0, 0] = move[1, 1] = move[2, 2] = scale
    move[3, :3] = pivot
    local = np.eye(4, dtype=np.float64)
    basis = world[:3, :3]
    for axis in range(3):
        length = float(np.linalg.norm(basis[axis]))
        if length < 1.0e-12:
            return []
        local[axis, :3] = basis[axis] / length * scale
    local[3, :3] = pivot
    return [move.reshape(-1).tolist(), local.reshape(-1).tolist(), local.reshape(-1).tolist()]


def _selection_outline_paths(
    render_path: str, render_shapes: list[dict[str, Any]]
) -> list[str]:
    prefix = str(render_path or "").rstrip("/")
    if prefix == "/BlacknodeOVRT/Ground":
        return ["/BlacknodeOVRT/Ground/Geometry"]
    descendants = [
        str(entry.get("path") or "")
        for entry in render_shapes
        if str(entry.get("path") or "") == prefix
        or str(entry.get("path") or "").startswith(prefix + "/")
    ]
    return list(dict.fromkeys(path for path in descendants if path)) or ([prefix] if prefix else [])


def _apply_selection_outline(
    renderer: Any,
    previous: list[str],
    selected: list[str],
    path_dictionary: Any = None,
) -> None:
    def assign(paths: list[str], group: int) -> None:
        if not paths:
            return
        if path_dictionary is None:
            renderer.set_selection_outline_group_strings(paths, [group] * len(paths))
            return
        path_ids = [int(path_dictionary.intern_path(path)) for path in paths]
        renderer.set_selection_outline_group(path_ids, group)

    assign(previous, 0)
    assign(selected, 1)


def _apply_collider_outline(
    renderer: Any,
    collider_paths: list[str],
    visible: bool,
    path_dictionary: Any = None,
) -> None:
    if not collider_paths:
        return
    group = 2 if visible else 0
    if path_dictionary is None:
        renderer.set_selection_outline_group_strings(
            collider_paths, [group] * len(collider_paths)
        )
        return
    path_ids = [int(path_dictionary.intern_path(path)) for path in collider_paths]
    renderer.set_selection_outline_group(path_ids, group)


def _load_environment_image(path: str) -> Any:
    """Load an HDR or EXR as linear RGB for viewport background compositing."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    import numpy as np

    source = Path(path)
    errors: list[str] = []
    image = None
    rgb_order = False
    # OpenCV wheels can advertise HDR support while being compiled without
    # OpenEXR. ImageIO's PyAV backend provides a reliable EXR fallback.
    if source.suffix.lower() == ".exr":
        try:
            import imageio.v3 as iio

            image = iio.imread(source)
            rgb_order = True
        except Exception as exc:
            errors.append(f"ImageIO: {exc}")
    if image is None:
        try:
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            errors.append(f"OpenCV: {exc}")
    if image is None and source.suffix.lower() == ".exr":
        try:
            import imageio_ffmpeg
            from PIL import Image

            command = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v", "error", "-i", str(source), "-frames:v", "1",
                "-f", "image2pipe", "-vcodec", "png", "-",
            ]
            options: dict[str, Any] = {
                "capture_output": True,
                "timeout": 30.0,
                "check": False,
            }
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(command, **options)
            if completed.returncode != 0 or not completed.stdout:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"ffmpeg exited with {completed.returncode}")
            image = np.asarray(Image.open(io.BytesIO(completed.stdout)).convert("RGB"))
            rgb_order = True
        except Exception as exc:
            errors.append(f"FFmpeg: {exc}")
    if image is None:
        detail = "; ".join(errors) or "no installed decoder accepted the file"
        raise RuntimeError(f"Could not decode HDRI {source}: {detail}")
    image = np.asarray(image)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 1:
        raise RuntimeError(f"HDRI decoder returned unsupported shape {image.shape}: {source}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    if not rgb_order:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        image = np.power(image.astype(np.float32) / maximum, 2.2)
    else:
        image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=65504.0, neginf=0.0)
    # Bound persistent float texture memory while retaining a sharp 2:1 dome
    # for the 1080p viewport. An 8K EXR can otherwise consume hundreds of MB in
    # both the decoder and the projected-background cache.
    maximum_width, maximum_height = 4096, 2048
    if image.shape[1] > maximum_width or image.shape[0] > maximum_height:
        scale = min(
            maximum_width / float(image.shape[1]),
            maximum_height / float(image.shape[0]),
        )
        image = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(image)


def _try_load_environment_image(path: str) -> Any:
    """Decode a viewport HDRI without allowing an optional codec to kill OVRT."""
    try:
        return _load_environment_image(path)
    except Exception as exc:
        _emit_status("rendering", f"HDRI lighting active; background preview unavailable: {exc}")
        return None


def _project_environment_background(
    camera: Any,
    image: Any,
    config: dict[str, Any],
    environment: dict[str, Any],
    *,
    interactive: bool = False,
) -> Any:
    if image is None:
        return None
    width = int(config.get("width") or 1)
    height = int(config.get("height") or 1)
    if not bool(environment.get("show_background", True)):
        import numpy as np

        color = _hex_color(
            str(environment.get("background_color") or "#6383c5")
        )
        pixel = np.asarray(
            [round(channel * 255.0) for channel in color], dtype=np.uint8
        )
        return np.broadcast_to(pixel, (height, width, 3)).copy()
    interaction_scale = (
        min(1.0, 640.0 / width, 360.0 / height) if interactive else 1.0
    )
    projection_width = max(1, int(round(width * interaction_scale)))
    projection_height = max(1, int(round(height * interaction_scale)))
    projected = camera.environment_background(
        image, projection_width, projection_height, _environment_intensity(environment)
    )
    if projection_width == width and projection_height == height:
        return projected
    import cv2

    # Camera drags use a half-resolution panorama projection for immediate
    # feedback. The settled frame is refined at full viewport resolution.
    return cv2.resize(projected, (width, height), interpolation=cv2.INTER_LINEAR)


def _matrix_array(matrices: list[list[float]]) -> Any:
    """Build an OVStage-compatible DLPack matrix payload."""
    import numpy as np

    return np.asarray(matrices, dtype=np.float64).reshape(-1)


def _vector_tensor(values: Any, lanes: int) -> Any:
    import numpy as np
    import ovstage

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    dtype = ovstage.numpy_to_dldatatype(flat.dtype, lanes=lanes)
    return ovstage.make_dltensor(flat, dtype=dtype, shape=[flat.size // lanes], ndim=1)


def _write_matrices(stage: Any, query: Any, xform_token: Any, ordinal: int, matrices: list[list[float]]) -> None:
    import numpy as np
    import ovstage

    if query is None or not matrices:
        return
    flat = _matrix_array(matrices)
    dtype = ovstage.numpy_to_dldatatype(flat.dtype, lanes=16)
    tensor = ovstage.make_dltensor(flat, dtype=dtype, shape=[len(matrices)], ndim=1)
    stage.write_attribute(
        query,
        xform_token,
        ordinal=ordinal,
        tensors=tensor,
        is_array=False,
        semantic=ovstage.AttributeSemantic.MATRIX,
    ).wait()


def _transforms_are_finite(transforms: Any) -> bool:
    """Reject incomplete or non-finite physics poses before they reach OVStage."""
    import numpy as np

    try:
        values = np.asarray(transforms, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[1] >= 7 and np.isfinite(values).all())


def _render_prim_path(path: str) -> str:
    return "/BlacknodeOVRT/Ground" if str(path) == "/Blacknode/Ground" else str(path)


def _render_shape_visible(
    entry: dict[str, Any],
    show_visuals: bool,
    show_colliders: bool,
    visibility_overrides: dict[str, bool] | None = None,
) -> bool:
    # Collider meshes are never surface-rendered. Their topology is drawn as a
    # green screen-space wireframe after RTX rendering, which guarantees that
    # the diagnostic overlay has higher Z priority than the visual geometry.
    category_visible = bool(entry.get("visual") and show_visuals)
    if not category_visible or not visibility_overrides:
        return category_visible
    path = str(entry.get("source_path") or entry.get("path") or "").rstrip("/")
    while path:
        if path in visibility_overrides and not visibility_overrides[path]:
            return False
        path = path.rsplit("/", 1)[0]
    return True


def _write_visibility(stage: Any, paths: Any, path: str, visible: bool, ordinal: int) -> None:
    """Publish one live USD visibility token through the attached OVStage."""
    import numpy as np
    import ovstage

    render_path = _render_prim_path(path)
    path_list = paths.create_path_list_from_strings([render_path])
    query = stage.query_from_path_list(path_list)
    try:
        visibility = paths.intern_token("visibility")
        value = paths.intern_token("inherited" if visible else "invisible")
        stage.write_attribute(
            query,
            visibility,
            ordinal=ordinal,
            tensors=np.asarray([value], dtype=np.uint64),
            is_array=False,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ).wait()
    finally:
        stage.release_query(query).wait()
        paths.destroy_path_list(path_list)


def _editor_matrix(transform: dict[str, Any], meters_per_unit: float) -> list[float]:
    from pxr import Gf

    value = Gf.Transform()
    translate = [float(item) for item in transform.get("translate_m", [0.0, 0.0, 0.0])]
    rotate = [float(item) for item in transform.get("rotate_deg", [0.0, 0.0, 0.0])]
    scale = [float(item) for item in transform.get("scale", [1.0, 1.0, 1.0])]
    units = max(1.0e-12, float(meters_per_unit))
    value.SetTranslation(Gf.Vec3d(*(item / units for item in translate)))
    value.SetRotation(
        Gf.Rotation(Gf.Vec3d.XAxis(), rotate[0])
        * Gf.Rotation(Gf.Vec3d.YAxis(), rotate[1])
        * Gf.Rotation(Gf.Vec3d.ZAxis(), rotate[2])
    )
    value.SetScale(Gf.Vec3d(*scale))
    return [float(item) for row in value.GetMatrix() for item in row]


def _query_for_path(stage: Any, paths: Any, path: str) -> tuple[Any, Any]:
    path_list = paths.create_path_list_from_strings([str(path)])
    return path_list, stage.query_from_path_list(path_list)


def _requires_renderer_reset(
    stage_updates: list[dict[str, Any]], visibility_updates: list[dict[str, Any]]
) -> bool:
    """Reset accumulated shading for infrequent discontinuous scene edits."""
    return bool(visibility_updates) or any(
        str(update.get("type") or "")
        in {"grid", "material", "environment", "render_options"}
        for update in stage_updates
    )


def _write_stage_update(stage: Any, paths: Any, update: dict[str, Any], ordinal: int) -> None:
    import numpy as np
    import ovstage

    kind = str(update.get("type") or "")
    if kind == "grid":
        for path in ("/BlacknodeOVRT/Grid", "/BlacknodeOVRT/GridAxes"):
            _write_visibility(stage, paths, path, bool(update.get("visible")), ordinal)
        return
    if kind == "transform":
        path_list, query = _query_for_path(
            stage, paths, _render_prim_path(str(update.get("path") or ""))
        )
        try:
            _write_matrices(
                stage,
                query,
                paths.intern_token("omni:xform"),
                ordinal,
                [_editor_matrix(
                    dict(update.get("transform") or {}),
                    float(update.get("meters_per_unit") or 1.0),
                )],
            )
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)
        return
    if kind == "material":
        prim_path = _render_prim_path(str(update.get("path") or ""))
        material_path = str(update.get("material_path") or "")
        material = dict(update.get("material") or {})
        shader_path = material_path + "/PreviewSurface"
        path_list, query = _query_for_path(stage, paths, shader_path)
        try:
            values = {
                "inputs:diffuseColor": _vector_tensor(material.get("base_color", [0.7] * 3), 3),
                "inputs:metallic": np.asarray([material.get("metallic", 0.0)], dtype=np.float32),
                "inputs:roughness": np.asarray([material.get("roughness", 0.5)], dtype=np.float32),
                "inputs:opacity": np.asarray([material.get("opacity", 1.0)], dtype=np.float32),
            }
            for name, tensor in values.items():
                stage.write_attribute(
                    query,
                    paths.intern_token(name),
                    ordinal=ordinal,
                    tensors=tensor,
                    is_array=False,
                    semantic=(
                        ovstage.AttributeSemantic.COLOR
                        if name == "inputs:diffuseColor"
                        else ovstage.AttributeSemantic.NONE
                    ),
                ).wait()
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)
        path_list, query = _query_for_path(stage, paths, prim_path)
        try:
            stage.write_attribute(
                query,
                paths.intern_token("material:binding"),
                ordinal=ordinal,
                tensors=np.asarray([paths.intern_path(material_path)], dtype=np.uint64),
                is_array=True,
                semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
            ).wait()
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)
        return
    if kind == "environment":
        environment = dict(update.get("environment") or {})
        hdri_path = _environment_texture_path(environment)
        distant = _distant_light(environment)
        key_paths, key_query = _query_for_path(
            stage, paths, "/BlacknodeOVRT/Key"
        )
        try:
            key_values = (
                (
                    "inputs:intensity",
                    np.asarray([
                        distant["intensity"] if distant["enabled"] else 0.0
                    ], dtype=np.float32),
                    ovstage.AttributeSemantic.NONE,
                ),
                (
                    "inputs:color",
                    _vector_tensor(distant["color"], 3),
                    ovstage.AttributeSemantic.VECTOR,
                ),
                (
                    "inputs:angle",
                    np.asarray([distant["angle_deg"]], dtype=np.float32),
                    ovstage.AttributeSemantic.NONE,
                ),
                (
                    "xformOp:rotateXYZ",
                    _vector_tensor(distant["rotation_deg"], 3),
                    ovstage.AttributeSemantic.VECTOR,
                ),
            )
            for name, tensor, semantic in key_values:
                stage.write_attribute(
                    key_query,
                    paths.intern_token(name),
                    ordinal=ordinal,
                    tensors=tensor,
                    is_array=False,
                    semantic=semantic,
                ).wait()
        finally:
            stage.release_query(key_query).wait()
            paths.destroy_path_list(key_paths)
        light_path = (
            "/BlacknodeOVRT/EnvironmentDome"
            if hdri_path
            else "/BlacknodeOVRT/Sky"
        )
        path_list, query = _query_for_path(stage, paths, light_path)
        try:
            color = _hex_color(str(environment.get("background_color") or "#6383c5"))
            light_color = (1.0, 1.0, 1.0) if hdri_path else color
            for name, tensor, semantic in (
                ("inputs:color", _vector_tensor(light_color, 3), ovstage.AttributeSemantic.VECTOR),
                ("inputs:intensity", np.asarray([_dome_light_intensity(environment)], dtype=np.float32), ovstage.AttributeSemantic.NONE),
            ):
                stage.write_attribute(
                    query, paths.intern_token(name), ordinal=ordinal, tensors=tensor,
                    is_array=False, semantic=semantic,
                ).wait()
            # The browser stream projects the source panorama at the camera's
            # full settled resolution. Keep the renderer dome out of primary
            # rays so background visibility never affects its illumination.
            stage.write_attribute(
                query,
                paths.intern_token("inputs:visibleInPrimaryRay"),
                ordinal=ordinal, tensors=np.asarray([False], dtype=np.bool_),
                is_array=False,
                semantic=ovstage.AttributeSemantic.NONE,
            ).wait()
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)
        # Exactly one dome contributes at a time. The authored environment
        # reference owns image-based lighting; the base dome owns solid color.
        base_paths, base_query = _query_for_path(
            stage, paths, "/BlacknodeOVRT/Sky"
        )
        try:
            base_intensity = (
                0.0 if hdri_path else _dome_light_intensity(environment)
            )
            stage.write_attribute(
                base_query,
                paths.intern_token("inputs:intensity"),
                ordinal=ordinal,
                tensors=np.asarray([base_intensity], dtype=np.float32),
                is_array=False,
                semantic=ovstage.AttributeSemantic.NONE,
            ).wait()
        finally:
            stage.release_query(base_query).wait()
            paths.destroy_path_list(base_paths)
        path_list, query = _query_for_path(stage, paths, RENDER_PRODUCT)
        try:
            stage.write_attribute(
                query,
                paths.intern_token("omni:rtx:background:source:type"),
                ordinal=ordinal,
                tensors=_background_source_tensor(environment),
                is_array=False,
                semantic=ovstage.AttributeSemantic.NONE,
            ).wait()
            stage.write_attribute(
                query,
                paths.intern_token("omni:rtx:background:source:color"),
                ordinal=ordinal,
                tensors=_vector_tensor(color, 3),
                is_array=False,
                semantic=ovstage.AttributeSemantic.COLOR,
            ).wait()
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)


def _decode_pick(products: Any, renderer: Any) -> dict[str, Any]:
    import numpy as np
    import ovrtx

    if RENDER_PRODUCT not in products:
        return {"path": ""}
    product = products[RENDER_PRODUCT]
    for frame in product.frames:
        if ovrtx.OVRTX_RENDER_VAR_PICK_HIT not in frame.render_vars:
            continue
        pick_var = frame.render_vars[ovrtx.OVRTX_RENDER_VAR_PICK_HIT]
        mapping = pick_var.map(device=ovrtx.Device.CPU)
        try:
            for name, expected in (
                ("magic", ovrtx.OVRTX_PICK_HIT_MAGIC),
                ("version", ovrtx.OVRTX_PICK_HIT_VERSION),
            ):
                if name in mapping.params:
                    actual = int(np.from_dlpack(mapping.params[name]).reshape(-1)[0])
                    if actual != int(expected):
                        return {"path": ""}
            hit_count = int(np.from_dlpack(mapping.params["hitCount"]).reshape(-1)[0])
            if hit_count <= 0:
                return {"path": ""}
            prim_paths = np.from_dlpack(mapping["primPath"]).copy().reshape(-1)
            world_positions = None
            try:
                world_positions = (
                    np.from_dlpack(mapping["worldPositionM"])
                    .copy()
                    .reshape(-1, 3)
                )
            except (KeyError, ValueError):
                pass
            for index, path_id in enumerate(prim_paths[:hit_count]):
                path = str(renderer.resolve_prim_path_id(int(path_id)) or "")
                if path:
                    result: dict[str, Any] = {"path": path}
                    if world_positions is not None and index < len(world_positions):
                        result["world_position_m"] = [
                            float(value) for value in world_positions[index]
                        ]
                    return result
        finally:
            mapping.unmap()
    return {"path": ""}


def _map_render_var(frame: Any, name: str) -> Any:
    import numpy as np
    import ovrtx

    if name not in frame.render_vars:
        return None
    mapped = frame.render_vars[name].map(device=ovrtx.Device.CPU)
    view = None
    try:
        view = np.from_dlpack(mapped)
        return view.copy()
    finally:
        if view is not None:
            del view
        mapped.unmap()
        del mapped


def _decode_semantic_id_map(tensor: Any) -> dict[int, str]:
    import numpy as np

    if tensor is None:
        return {}
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    if data.size < 4:
        return {}
    entry_dtype = np.dtype([("id", "<u4", (4)), ("label_length", "<u4"), ("label_offset", "<u4")])
    count = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    entries_size = count * entry_dtype.itemsize
    if count < 0 or entries_size > data.size - 4:
        return {}
    result: dict[int, str] = {}
    for entry in data[:entries_size].view(entry_dtype).reshape(count):
        semantic_id = int(entry["id"][0])
        offset = int(entry["label_offset"])
        length = int(entry["label_length"])
        if offset < 0 or length < 0 or offset + length > data.size:
            continue
        result[semantic_id] = data[offset:offset + length].tobytes().decode(
            "utf-8", errors="replace"
        ).rstrip("\x00").rstrip()
    return result


def _semantic_color(semantic_id: int) -> tuple[int, int, int]:
    """Stable high-contrast color used by segmentation and detection overlays."""
    import colorsys

    hue = (int(semantic_id) * 0.618033988749895 + 0.08) % 1.0
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.78, 1.0))


def _infrared_depth(distance: Any) -> Any:
    import numpy as np

    values = np.squeeze(distance).astype(np.float32, copy=False)
    valid = np.isfinite(values) & (values > 0.0)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output
    near, far = np.percentile(values[valid], [2.0, 98.0])
    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        far = near + 1.0
    heat = 1.0 - np.clip((values - near) / (far - near), 0.0, 1.0)
    stops = np.asarray([0.0, 0.2, 0.45, 0.7, 0.88, 1.0], dtype=np.float32)
    colors = np.asarray([
        [5, 0, 20], [40, 8, 95], [155, 18, 95],
        [235, 55, 35], [255, 185, 25], [255, 255, 220],
    ], dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(heat, stops, colors[:, channel]).astype(np.uint8)
    output[~valid] = 0
    return output


def _semantic_label(raw: str, semantic_id: int) -> str:
    fields: dict[str, str] = {}
    for part in str(raw or "").split(";"):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        fields[name.strip()] = value.strip()
    return fields.get("label") or fields.get("class") or f"object {semantic_id}"


def _semantic_visuals(ids: Any, labels: dict[int, str]) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    values = np.squeeze(ids).astype(np.uint32, copy=False)
    color = np.zeros((*values.shape, 3), dtype=np.uint8)
    boxes: list[dict[str, Any]] = []
    image_area = max(1, values.shape[0] * values.shape[1])
    for raw_id in np.unique(values):
        semantic_id = int(raw_id)
        if semantic_id == 0:
            continue
        if labels and semantic_id not in labels:
            continue
        mask = values == semantic_id
        count = int(np.count_nonzero(mask))
        if count < 12:
            continue
        rgb = _semantic_color(semantic_id)
        color[mask] = rgb
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if (x1 - x0 + 1) * (y1 - y0 + 1) > image_area * 0.98:
            continue
        boxes.append({
            "id": semantic_id,
            "label": _semantic_label(labels.get(semantic_id, ""), semantic_id),
            "color": rgb,
            "bounds": (x0, y0, x1, y1),
            "pixels": count,
        })
    return color, boxes


def _draw_detection_boxes(image: Any, boxes: list[dict[str, Any]]) -> Any:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in sorted(boxes, key=lambda item: int(item["pixels"]), reverse=True):
        color = tuple(box["color"])
        bounds = tuple(box["bounds"])
        draw.rectangle(bounds, outline=color, width=3)
        label = str(box["label"])
        text_box = draw.textbbox((bounds[0], bounds[1]), label, font=font, stroke_width=1)
        width = max(1, text_box[2] - text_box[0] + 8)
        height = max(1, text_box[3] - text_box[1] + 6)
        y = max(0, bounds[1] - height)
        draw.rectangle((bounds[0], y, bounds[0] + width, y + height), fill=color)
        draw.text((bounds[0] + 4, y + 3), label, fill=(7, 9, 13), font=font, stroke_width=0)
    return image


def _semantic_key(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _selection_semantic_mask(ids: Any, labels: dict[int, str], selected_path: str) -> Any:
    import numpy as np

    values = np.squeeze(ids).astype(np.uint32, copy=False)
    selected = str(selected_path or "").rstrip("/")
    leaf_key = _semantic_key(selected.rsplit("/", 1)[-1])
    path_key = _semantic_key(selected)
    if not leaf_key:
        return None
    exact_path_ids = [
        semantic_id
        for semantic_id, raw in labels.items()
        if path_key and path_key in _semantic_key(raw)
    ]
    matched_ids = exact_path_ids or [
        semantic_id
        for semantic_id, raw in labels.items()
        if _semantic_key(_semantic_label(raw, semantic_id)) == leaf_key
    ]
    if not matched_ids:
        return None
    mask = np.isin(values, np.asarray(matched_ids, dtype=np.uint32))
    return mask if np.any(mask) else None


def _draw_selection_outline(
    image: Any, ids: Any, labels: dict[int, str], selected_path: str
) -> Any:
    from PIL import Image, ImageChops, ImageFilter

    mask = _selection_semantic_mask(ids, labels, selected_path)
    if mask is None:
        return image
    source = Image.fromarray(mask.astype("uint8") * 255, mode="L")
    outer = source.filter(ImageFilter.MaxFilter(5))
    inner = source.filter(ImageFilter.MinFilter(3))
    edge = ImageChops.subtract(outer, inner).point(lambda value: round(value * 0.82))
    color = Image.new("RGB", image.size, (255, 199, 56))
    return Image.composite(color, image.convert("RGB"), edge)


def _draw_collision_wireframes(
    image: Any,
    wireframes: list[dict[str, Any]],
    transforms: list[list[float]],
    camera: Camera,
    meters_per_unit: float,
) -> Any:
    """Composite collision edges after RTX so they remain visible through geometry."""
    import cv2
    import numpy as np
    from PIL import Image

    if not wireframes:
        return image
    canvas = np.asarray(image.convert("RGB")).copy()
    height, width = canvas.shape[:2]
    forward = camera.target - camera.eye
    forward_length = float(np.linalg.norm(forward))
    if forward_length < 1.0e-9:
        return image
    forward /= forward_length
    world_up = np.zeros(3, dtype=np.float64)
    world_up[{"x": 0, "y": 1, "z": 2}.get(camera.up_axis, 2)] = 1.0
    right = np.cross(forward, world_up)
    right_length = float(np.linalg.norm(right))
    if right_length < 1.0e-9:
        return image
    right /= right_length
    up = np.cross(right, forward)
    tan_half_horizontal = 36.0 / (2.0 * 32.0)
    tan_half_vertical = tan_half_horizontal * height / max(1, width)
    line_batches: list[Any] = []
    observed_depths: list[Any] = []
    for entry in wireframes:
        points = np.asarray(entry.get("points_bind_world") or [], dtype=np.float64)
        edges = np.asarray(entry.get("edges") or [], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3 or edges.ndim != 2:
            continue
        body_index = int(entry.get("body_index", -1))
        body_bind_values = list(entry.get("body_bind_world_matrix") or [])
        if (
            body_index >= 0
            and body_index < len(transforms)
            and len(body_bind_values) == 16
        ):
            body_bind = np.asarray(body_bind_values, dtype=np.float64).reshape(4, 4)
            body_current = np.asarray(
                _pose_matrix(
                    transforms[body_index], [1.0, 1.0, 1.0], meters_per_unit
                ),
                dtype=np.float64,
            ).reshape(4, 4)
            homogeneous = np.column_stack((points, np.ones(len(points))))
            points = (homogeneous @ np.linalg.inv(body_bind) @ body_current)[:, :3]
        offsets = points - camera.eye
        depths = offsets @ forward
        observed_depths.append(depths)
        valid_points = depths > 1.0e-6
        safe_depths = np.maximum(depths, 1.0e-6)
        screen = np.column_stack((
            (0.5 + (offsets @ right) / (2.0 * safe_depths * tan_half_horizontal)) * width,
            (0.5 - (offsets @ up) / (2.0 * safe_depths * tan_half_vertical)) * height,
        ))
        valid_edges = valid_points[edges[:, 0]] & valid_points[edges[:, 1]]
        if not np.any(valid_edges):
            continue
        segments = np.rint(screen[edges[valid_edges]]).astype(np.int32)
        # Drop extreme off-screen coordinates before handing contours to OpenCV.
        coordinate_limit = max(width, height) * 8
        finite = np.all(np.abs(segments) <= coordinate_limit, axis=(1, 2))
        line_batches.extend(segments[finite])
    if line_batches:
        cv2.polylines(
            canvas,
            np.asarray(line_batches, dtype=np.int32).reshape(-1, 2, 2),
            False,
            (51, 255, 82),
            2,
            lineType=cv2.LINE_AA,
        )
    with STATE.lock:
        STATE.collision_overlay_segments = len(line_batches)
        if observed_depths:
            all_depths = np.concatenate(observed_depths)
            STATE.collision_depth_range = [
                float(np.min(all_depths)), float(np.max(all_depths))
            ]
        STATE.collision_overlay_pixels = int(
            np.count_nonzero(
                (canvas[:, :, 1] > 180)
                & (canvas[:, :, 1] > canvas[:, :, 0] * 1.4)
                & (canvas[:, :, 1] > canvas[:, :, 2] * 1.4)
            )
        )
    return Image.fromarray(canvas)


def _encode_frame(
    products: Any,
    quality: int,
    view_mode: str = "rgb",
    environment_background: Any = None,
    selected_path: str = "",
    collision_wireframes: list[dict[str, Any]] | None = None,
    collision_transforms: list[list[float]] | None = None,
    collision_camera: Camera | None = None,
    meters_per_unit: float = 1.0,
    show_colliders: bool = False,
) -> bytes:
    import numpy as np

    for product in products.values():
        for frame in product.frames:
            from PIL import Image

            pixels = _map_render_var(frame, "LdrColor")
            if pixels is None:
                continue
            rgb = pixels[..., :3].copy()
            semantic_ids = None
            semantic_labels: dict[int, str] = {}
            if environment_background is not None:
                distance_for_background = _map_render_var(frame, "DistanceToCameraSD")
                if distance_for_background is not None:
                    distance_values = np.squeeze(distance_for_background)
                    background_mask = ~np.isfinite(distance_values) | (distance_values <= 0.0)
                    if background_mask.shape == rgb.shape[:2]:
                        rgb[background_mask] = environment_background[background_mask]
            mode = str(view_mode or "rgb").lower()
            if mode == "depth":
                distance = _map_render_var(frame, "DistanceToCameraSD")
                image = Image.fromarray(_infrared_depth(distance)) if distance is not None else Image.fromarray(rgb)
            elif mode in {"segmentation", "detection", "composite"}:
                semantic_ids = _map_render_var(frame, "SemanticSegmentation")
                label_tensor = _map_render_var(frame, "SemanticIdMap")
                semantic_labels = _decode_semantic_id_map(label_tensor)
                if semantic_ids is None:
                    image = Image.fromarray(rgb)
                else:
                    segmentation, boxes = _semantic_visuals(semantic_ids, semantic_labels)
                    if mode == "segmentation":
                        image = Image.fromarray(segmentation)
                    elif mode == "composite":
                        base = Image.fromarray(rgb).convert("RGB")
                        overlay = Image.fromarray(segmentation).convert("RGB")
                        image = Image.blend(base, overlay, 0.42)
                        image = _draw_detection_boxes(image, boxes)
                    else:
                        image = _draw_detection_boxes(Image.fromarray(rgb).convert("RGB"), boxes)
            else:
                image = Image.fromarray(rgb)
            if selected_path:
                if semantic_ids is None:
                    semantic_ids = _map_render_var(frame, "SemanticSegmentation")
                    semantic_labels = _decode_semantic_id_map(
                        _map_render_var(frame, "SemanticIdMap")
                    )
                if semantic_ids is not None:
                    image = _draw_selection_outline(
                        image, semantic_ids, semantic_labels, selected_path
                    )
            if show_colliders and collision_wireframes and collision_camera is not None:
                image = _draw_collision_wireframes(
                    image,
                    list(collision_wireframes or []),
                    list(collision_transforms or []),
                    collision_camera,
                    meters_per_unit,
                )
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=False)
            return output.getvalue()
    # Empty products are expected while OVRT is compiling a new shader set.
    return b""


def render(config: dict[str, Any]) -> None:
    import numpy as np
    import ovrtx
    import ovstage

    with STATE.lock:
        STATE.phase = "initializing"
        STATE.detail = "Creating NVIDIA OVRT renderer; first launch compiles RTX shaders"
    renderer = None
    stage = None
    camera = Camera(config)
    body_entries = list(config.get("body_entries") or [])
    initial_body_transforms = list(config.get("initial_body_transforms") or [])
    ghost_entries = list(config.get("ghost_entries") or [])
    render_shapes = list(config.get("render_shapes") or [])
    collision_wireframes = list(config.get("collision_wireframes") or [])
    current_transforms = list(initial_body_transforms)
    render_shapes_by_path = {
        str(entry.get("path") or ""): entry for entry in render_shapes
    }
    display_proxy_source_paths = {
        str(entry.get("source_path") or "")
        for entry in render_shapes
        if entry.get("visual")
        and str(entry.get("source_path") or "")
        and str(entry.get("source_path") or "") != str(entry.get("path") or "")
    }
    bound_shape_entries = [
        entry
        for entry in render_shapes
        if entry.get("visual")
        and int(entry.get("body_index", -1)) >= 0
        and len(list(entry.get("initial_world_matrix") or [])) == 16
        and len(list(entry.get("body_bind_world_matrix") or [])) == 16
    ]
    collider_outline_paths = list(dict.fromkeys(
        str(entry.get("path") or "")
        for entry in render_shapes
        if entry.get("collider") and str(entry.get("path") or "")
    ))
    interaction_frames = list(config.get("interaction_frames") or [])
    interaction_frames_by_render_path = {
        str(entry.get("render_path") or ""): entry for entry in interaction_frames
    }
    interaction_frames_by_path = {
        str(entry.get("path") or ""): entry for entry in interaction_frames
    }
    visibility_overrides = {
        str(path): bool(visible)
        for path, visible in dict(config.get("visibility_overrides") or {}).items()
    }
    show_visuals = bool(config.get("show_visuals", True))
    show_colliders = bool(config.get("show_colliders", False))
    with STATE.lock:
        STATE.collision_wireframe_count = len(collision_wireframes)
        STATE.colliders_visible = show_colliders
    transform_overrides: dict[str, list[float]] = {}
    selected_frame: dict[str, Any] | None = None
    selected_render_path = ""
    gizmo_tool = "select"
    gizmo_visibility = {"move": False, "rotate": False, "scale": False}
    last_native_gizmo_matrices: list[list[float]] = []
    outlined_paths: list[str] = []
    environment = dict(config.get("environment") or {})
    hdri_path = _environment_texture_path(environment)
    environment_image = (
        _try_load_environment_image(hdri_path) if hdri_path else None
    )
    environment_background = _project_environment_background(
        camera, environment_image, config, environment
    )
    environment_background_refine_at = 0.0
    environment_dome_handle: int | None = None
    environment_dome_path = ""
    ordinal = 1
    try:
        _emit_status("initializing", "Creating NVIDIA OVRT renderer; first launch compiles RTX shaders")
        renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(
            selection_outline_enabled=True,
            selection_outline_width=3,
            selection_fill_mode=ovrtx.SelectionFillMode.EDGE_ONLY,
        ))
        _start_input_reader()
        _emit_status("loading", "OVRT renderer ready; attaching OVStage")
        stage = ovstage.Stage("blacknode.newton.ovrtx")
        renderer.attach_ovstage(stage)
        wrapper = _wrapper_usda(config, camera.matrix())
        with contextlib.nullcontext():
            with STATE.lock:
                STATE.detail = "Loading USD into OVStage"
            _emit_status("loading", "Loading USD into OVStage")
            ovstage.population.open_usd_from_string(
                stage,
                wrapper,
                ordinal=ordinal,
                time_code=0.0,
                domains=ovstage.PopulationDomain.RENDERING,
            )
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            if hdri_path:
                ordinal += 1
                environment_dome_handle = _replace_environment_dome(
                    stage,
                    environment_dome_handle,
                    hdri_path,
                    environment,
                    ordinal,
                )
                environment_dome_path = hdri_path
                stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            renderer.reset()
            renderer.set_selection_group_styles({
                1: ovrtx.SelectionGroupStyle(
                    outline_color=(1.0, 0.78, 0.22, 1.0),
                    fill_color=(1.0, 0.78, 0.22, 1.0),
                ),
                2: ovrtx.SelectionGroupStyle(
                    outline_color=(0.20, 1.0, 0.32, 1.0),
                    fill_color=(0.20, 1.0, 0.32, 0.0),
                ),
            })

            with ovstage.PathDictionary(stage) as paths:
                xform = paths.intern_token("omni:xform")
                camera_paths = paths.create_path_list_from_strings(["/BlacknodeOVRT/Camera"])
                camera_query = stage.query_from_path_list(camera_paths)
                gizmo_root_paths = [
                    "/BlacknodeOVRT/GizmoMove",
                    "/BlacknodeOVRT/GizmoRotate",
                    "/BlacknodeOVRT/GizmoScale",
                ]
                gizmo_paths = paths.create_path_list_from_strings(gizmo_root_paths)
                gizmo_query = stage.query_from_path_list(gizmo_paths)
                body_paths = None
                body_query = None
                if body_entries:
                    body_paths = paths.create_path_list_from_strings([str(entry["path"]) for entry in body_entries])
                    body_query = stage.query_from_path_list(body_paths)
                bound_shape_paths = None
                bound_shape_query = None
                if bound_shape_entries:
                    bound_shape_paths = paths.create_path_list_from_strings(
                        [
                            str(entry.get("transform_path") or entry["path"])
                            for entry in bound_shape_entries
                        ]
                    )
                    bound_shape_query = stage.query_from_path_list(bound_shape_paths)
                ghost_paths = None
                ghost_query = None
                if ghost_entries:
                    ghost_paths = paths.create_path_list_from_strings(
                        [str(entry["path"]) for entry in ghost_entries]
                    )
                    ghost_query = stage.query_from_path_list(ghost_paths)
                try:
                    if render_shapes:
                        ordinal += 1
                        for entry in render_shapes:
                            _write_visibility(
                                stage,
                                paths,
                                str(entry.get("path") or ""),
                                _render_shape_visible(
                                    entry,
                                    show_visuals,
                                    show_colliders,
                                    visibility_overrides,
                                ),
                                ordinal,
                            )
                        _apply_collider_outline(
                            renderer, collider_outline_paths, show_colliders, paths
                        )
                        if bound_shape_entries and current_transforms:
                            initial_shape_matrices = [
                                _bound_shape_pose_matrix(
                                    entry,
                                    current_transforms,
                                    float(config.get("meters_per_unit") or 1.0),
                                )
                                for entry in bound_shape_entries
                            ]
                            _write_matrices(
                                stage,
                                bound_shape_query,
                                xform,
                                ordinal,
                                initial_shape_matrices,
                            )
                        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                    warmup = max(1, int(config.get("warmup_frames") or 1))
                    ready = False
                    for index in range(warmup):
                        if STATE.stop:
                            return
                        with STATE.lock:
                            STATE.detail = f"Compiling RTX shaders · pass {index + 1}/{warmup}"
                        _emit_status("warming", f"Compiling RTX shaders · pass {index + 1}/{warmup}")
                        products = renderer.step(
                            render_products={RENDER_PRODUCT}, delta_time=1.0 / 60.0, ordinal=ordinal
                        )
                        jpeg = _encode_frame(
                            products,
                            int(config["jpeg_quality"]),
                            "rgb",
                            environment_background,
                            "",
                            collision_wireframes,
                            current_transforms,
                            camera,
                            float(config.get("meters_per_unit") or 1.0),
                            show_colliders,
                        )
                        del products
                        if jpeg:
                            STATE.publish(jpeg)
                            ready = True
                            break
                    if not ready:
                        raise RuntimeError(
                            f"OVRT produced no LdrColor frame after {warmup} initialization passes"
                        )
                    _emit_status(
                        "streaming",
                        "OVRT RTX render stream · "
                        f"{len(collision_wireframes)} collision wireframe layer(s) · "
                        f"{'visible' if show_colliders else 'hidden'}",
                    )

                    frame_interval = 1.0 / float(config.get("render_fps") or 60)
                    while not STATE.stop:
                        started = time.monotonic()
                        with STATE.lock:
                            transforms = STATE.latest_transforms
                            STATE.latest_transforms = None
                            reference = STATE.latest_reference
                            STATE.latest_reference = None
                            actions = STATE.camera_actions[:]
                            STATE.camera_actions.clear()
                            visibility_updates = STATE.visibility_updates[:]
                            STATE.visibility_updates.clear()
                            stage_updates = STATE.stage_updates[:]
                            STATE.stage_updates.clear()
                            pick_requests = STATE.pick_requests[:]
                            STATE.pick_requests.clear()
                            selection_requests = STATE.selection_requests[:]
                            STATE.selection_requests.clear()
                            view_mode = STATE.view_mode
                        if transforms is not None and not _transforms_are_finite(transforms):
                            # Keep the last complete pose instead of poisoning
                            # render matrices with NaNs from an unstable step.
                            transforms = None
                        if transforms is not None:
                            current_transforms = list(transforms)
                        for visibility_update in visibility_updates:
                            path = str(visibility_update.get("path") or "")
                            requested = bool(visibility_update.get("visible"))
                            visibility_overrides[path] = requested
                            # A source Gprim remains permanently surface-hidden;
                            # its live standalone proxy carries user visibility.
                            visibility_update["skip_render_write"] = (
                                path in display_proxy_source_paths
                            )
                            entry = render_shapes_by_path.get(path)
                            if entry is not None:
                                visibility_update["visible"] = _render_shape_visible(
                                    entry,
                                    show_visuals,
                                    show_colliders,
                                    visibility_overrides,
                                )
                            # Collision display proxies live outside the source
                            # hierarchy, so mirror a parent/source visibility
                            # edit onto every proxy representing that subtree.
                            for candidate in render_shapes:
                                source_path = str(candidate.get("source_path") or "")
                                candidate_path = str(candidate.get("path") or "")
                                if not source_path or source_path == candidate_path:
                                    continue
                                if source_path == path or source_path.startswith(path.rstrip("/") + "/"):
                                    visibility_updates.append({
                                        "path": candidate_path,
                                        "visible": _render_shape_visible(
                                            candidate,
                                            show_visuals,
                                            show_colliders,
                                            visibility_overrides,
                                        ),
                                    })
                        camera_dirty = camera.apply(actions)
                        if (
                            camera_dirty
                            and environment_image is not None
                            and bool(environment.get("show_background", True))
                        ):
                            environment_background = _project_environment_background(
                                camera,
                                environment_image,
                                config,
                                environment,
                                interactive=True,
                            )
                            environment_background_refine_at = time.monotonic() + 0.12
                        elif (
                            environment_background_refine_at > 0.0
                            and time.monotonic() >= environment_background_refine_at
                        ):
                            environment_background = _project_environment_background(
                                camera, environment_image, config, environment
                            )
                            environment_background_refine_at = 0.0
                        if selection_requests:
                            selected_path = str(selection_requests[-1] or "")
                            selected_frame = interaction_frames_by_path.get(selected_path)
                            selected_render_path = str(
                                (selected_frame or {}).get("render_path") or selected_path
                            )
                            new_outline_paths = _selection_outline_paths(
                                selected_render_path, render_shapes
                            )
                            _apply_selection_outline(
                                renderer, outlined_paths, new_outline_paths, paths
                            )
                            _apply_collider_outline(
                                renderer, collider_outline_paths, show_colliders, paths
                            )
                            _apply_selection_outline(
                                renderer, [], new_outline_paths, paths
                            )
                            outlined_paths = new_outline_paths
                            gizmo = _selection_gizmo(
                                camera,
                                selected_frame,
                                config,
                                transform_overrides,
                                body_transforms=transforms,
                            )
                            if STREAM_TO_STDOUT:
                                _send_event({
                                    "type": "selection",
                                    "path": selected_path,
                                    "gizmo": gizmo,
                                })
                        for stage_update in stage_updates:
                            if stage_update.get("type") == "transform":
                                update_path = str(stage_update.get("path") or "")
                                if bool(stage_update.get("physics_backed")):
                                    transform_overrides.pop(update_path, None)
                                else:
                                    transform_overrides[update_path] = _editor_matrix(
                                        dict(stage_update.get("transform") or {}),
                                        float(stage_update.get("meters_per_unit") or 1.0),
                                    )
                            elif stage_update.get("type") == "environment":
                                environment = dict(stage_update.get("environment") or {})
                                hdri_path = _environment_texture_path(environment)
                                environment_image = (
                                    _try_load_environment_image(hdri_path)
                                    if hdri_path
                                    else None
                                )
                                environment_background = _project_environment_background(
                                    camera, environment_image, config, environment
                                )
                                environment_background_refine_at = 0.0
                            elif stage_update.get("type") == "render_options":
                                show_visuals = bool(stage_update.get("show_visuals", True))
                                show_colliders = bool(stage_update.get("show_colliders", False))
                                with STATE.lock:
                                    STATE.colliders_visible = show_colliders
                                visibility_updates.extend({
                                    "path": str(entry.get("path") or ""),
                                    "visible": _render_shape_visible(
                                        entry,
                                        show_visuals,
                                        show_colliders,
                                        visibility_overrides,
                                    ),
                                } for entry in render_shapes)
                                _apply_collider_outline(
                                    renderer, collider_outline_paths, show_colliders, paths
                                )
                                if outlined_paths:
                                    _apply_selection_outline(
                                        renderer, [], outlined_paths, paths
                                    )
                            elif stage_update.get("type") == "gizmo_tool":
                                requested_tool = str(stage_update.get("tool") or "select")
                                if requested_tool in {"select", "move", "rotate", "scale"}:
                                    gizmo_tool = requested_tool
                        body_matrices: list[list[float]] = []
                        if transforms is not None:
                            for entry in body_entries:
                                index = int(entry["index"])
                                if index < len(transforms):
                                    body_matrices.append(
                                        transform_overrides.get(str(entry.get("path") or ""))
                                        or _body_pose_matrix(
                                            entry,
                                            transforms,
                                            float(config.get("meters_per_unit") or 1.0),
                                        )
                                    )
                        bound_shape_matrices: list[list[float]] = []
                        if transforms is not None:
                            for entry in bound_shape_entries:
                                body_index = int(entry["body_index"])
                                if body_index < len(transforms):
                                    bound_shape_matrices.append(
                                        _bound_shape_pose_matrix(
                                            entry,
                                            transforms,
                                            float(config.get("meters_per_unit") or 1.0),
                                            _interaction_transform_delta(
                                                str(entry.get("path") or ""),
                                                interaction_frames_by_path,
                                                transform_overrides,
                                            ),
                                        )
                                    )
                        native_gizmo_matrices = (
                            _native_gizmo_matrices(
                                camera,
                                selected_frame,
                                config,
                                transform_overrides,
                                current_transforms,
                            )
                            if selected_frame is not None and gizmo_tool != "select"
                            else []
                        )
                        desired_gizmo_visibility = {
                            name: bool(
                                native_gizmo_matrices
                                and gizmo_tool == name
                                and (name != "scale" or selected_frame.get("scale_editable", True))
                            )
                            for name in ("move", "rotate", "scale")
                        }
                        gizmo_visibility_updates = [
                            (name, visible)
                            for name, visible in desired_gizmo_visibility.items()
                            if gizmo_visibility.get(name) != visible
                        ]
                        gizmo_matrix_dirty = bool(
                            native_gizmo_matrices
                            and (
                                len(native_gizmo_matrices) != len(last_native_gizmo_matrices)
                                or any(
                                    abs(float(value) - float(previous)) > 1.0e-9
                                    for matrix, old_matrix in zip(
                                        native_gizmo_matrices, last_native_gizmo_matrices
                                    )
                                    for value, previous in zip(matrix, old_matrix)
                                )
                            )
                        )
                        ghost_matrices: list[list[float]] = []
                        ghost_visible: bool | None = None
                        if reference is not None and ghost_entries:
                            reference_transforms = list(reference.get("transforms") or [])
                            offset = list(reference.get("offset_m") or [0.0, 0.0, 0.0])
                            offset.extend([0.0] * (3 - len(offset)))
                            shifted_transforms: list[list[float]] = []
                            for transform in reference_transforms:
                                shifted = list(transform)
                                if len(shifted) >= 3:
                                    shifted[0] = float(shifted[0]) + float(offset[0])
                                    shifted[1] = float(shifted[1]) + float(offset[1])
                                    shifted[2] = float(shifted[2]) + float(offset[2])
                                shifted_transforms.append(shifted)
                            for entry in ghost_entries:
                                index = int(entry["index"])
                                if index < len(shifted_transforms):
                                    ghost_matrices.append(_body_pose_matrix(
                                        entry,
                                        shifted_transforms,
                                        float(config.get("meters_per_unit") or 1.0),
                                    ))
                            ghost_visible = bool(
                                reference.get("visible", True)
                                and len(ghost_matrices) == len(ghost_entries)
                            )
                        if (
                            camera_dirty
                            or body_matrices
                            or bound_shape_matrices
                            or reference is not None
                            or visibility_updates
                            or stage_updates
                            or gizmo_matrix_dirty
                            or gizmo_visibility_updates
                        ):
                            ordinal += 1
                            if camera_dirty:
                                _write_matrices(stage, camera_query, xform, ordinal, [camera.matrix()])
                            if gizmo_matrix_dirty:
                                _write_matrices(
                                    stage, gizmo_query, xform, ordinal, native_gizmo_matrices
                                )
                                last_native_gizmo_matrices = [
                                    list(matrix) for matrix in native_gizmo_matrices
                                ]
                            if body_matrices and len(body_matrices) == len(body_entries):
                                _write_matrices(stage, body_query, xform, ordinal, body_matrices)
                            if bound_shape_matrices and len(bound_shape_matrices) == len(bound_shape_entries):
                                _write_matrices(
                                    stage,
                                    bound_shape_query,
                                    xform,
                                    ordinal,
                                    bound_shape_matrices,
                                )
                            if ghost_matrices and len(ghost_matrices) == len(ghost_entries):
                                _write_matrices(stage, ghost_query, xform, ordinal, ghost_matrices)
                            if ghost_visible is not None:
                                for entry in ghost_entries:
                                    _write_visibility(
                                        stage,
                                        paths,
                                        str(entry["path"]),
                                        ghost_visible,
                                        ordinal,
                                    )
                            for visibility_update in visibility_updates:
                                if visibility_update.get("skip_render_write"):
                                    continue
                                _write_visibility(
                                    stage,
                                    paths,
                                    str(visibility_update["path"]),
                                    bool(visibility_update["visible"]),
                                    ordinal,
                                )
                            for name, visible in gizmo_visibility_updates:
                                _write_visibility(
                                    stage,
                                    paths,
                                    f"/BlacknodeOVRT/Gizmo{name.title()}",
                                    visible,
                                    ordinal,
                                )
                                gizmo_visibility[name] = visible
                            for stage_update in stage_updates:
                                try:
                                    if stage_update.get("type") == "environment":
                                        requested_environment = dict(
                                            stage_update.get("environment") or {}
                                        )
                                        requested_path = _environment_texture_path(
                                            requested_environment
                                        )
                                        if requested_path != environment_dome_path:
                                            environment_dome_handle = _replace_environment_dome(
                                                stage,
                                                environment_dome_handle,
                                                requested_path,
                                                requested_environment,
                                                ordinal,
                                            )
                                            environment_dome_path = requested_path
                                    if not (
                                        stage_update.get("type") == "transform"
                                        and bool(stage_update.get("physics_backed"))
                                    ):
                                        _write_stage_update(stage, paths, stage_update, ordinal)
                                except Exception as exc:
                                    # A live material/environment column can be
                                    # rejected by a particular OVStage release.
                                    # Keep the renderer and last valid frame
                                    # alive instead of terminating the window.
                                    kind = str(stage_update.get("type") or "stage")
                                    _emit_status(
                                        "streaming",
                                        f"Live {kind} update rejected; viewer kept active: {exc}",
                                    )
                            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                            # Discontinuous shading/visibility changes need a
                            # clean accumulation buffer. Camera and transform
                            # drags stay on lightweight matrix writes so orbit
                            # and gizmos do not repeatedly reset the renderer.
                            if _requires_renderer_reset(
                                stage_updates, visibility_updates
                            ) or gizmo_visibility_updates:
                                renderer.reset()
                        if pick_requests:
                            pick = pick_requests[-1]
                            pixel_x = 1.0 / max(1, int(config.get("width") or 1))
                            pixel_y = 1.0 / max(1, int(config.get("height") or 1))
                            left = min(1.0 - pixel_x, max(0.0, float(pick["x"])))
                            top = min(1.0 - pixel_y, max(0.0, float(pick["y"])))
                            renderer.enqueue_pick_query(
                                render_product_path=RENDER_PRODUCT,
                                left_ndc=left,
                                top_ndc=top,
                                right_ndc=left + pixel_x,
                                bottom_ndc=top + pixel_y,
                                flags=ovrtx.OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO,
                            )
                        products = renderer.step(
                            render_products={RENDER_PRODUCT}, delta_time=frame_interval, ordinal=ordinal
                        )
                        jpeg = _encode_frame(
                            products,
                            int(config["jpeg_quality"]),
                            view_mode,
                            environment_background,
                            selected_render_path,
                            collision_wireframes,
                            current_transforms,
                            camera,
                            float(config.get("meters_per_unit") or 1.0),
                            show_colliders,
                        )
                        if jpeg:
                            STATE.publish(jpeg)
                        if pick_requests:
                            hit = _decode_pick(products, renderer)
                            raw_selected = str(hit.get("path") or "")
                            if raw_selected.startswith("/BlacknodeOVRT/Gizmo"):
                                raw_selected = selected_render_path
                            selected_frame = _resolve_interaction_frame(
                                raw_selected, interaction_frames_by_render_path
                            )
                            selected_path = str(
                                (selected_frame or {}).get("path") or raw_selected
                            )
                            selected_render_path = raw_selected
                            new_outline_paths = _selection_outline_paths(
                                raw_selected, render_shapes
                            )
                            try:
                                _apply_selection_outline(
                                    renderer, outlined_paths, new_outline_paths, paths
                                )
                                _apply_collider_outline(
                                    renderer, collider_outline_paths, show_colliders, paths
                                )
                                _apply_selection_outline(
                                    renderer, [], new_outline_paths, paths
                                )
                                outlined_paths = new_outline_paths
                            except (AttributeError, RuntimeError):
                                pass
                            gizmo = _selection_gizmo(
                                camera,
                                selected_frame,
                                config,
                                transform_overrides,
                                [float(pick["x"]), float(pick["y"])],
                                current_transforms,
                            )
                            if STREAM_TO_STDOUT:
                                _send_event({
                                    "type": "selection",
                                    "path": selected_path,
                                    "gizmo": gizmo,
                                })
                        elif (camera_dirty or stage_updates) and selected_frame is not None:
                            if STREAM_TO_STDOUT:
                                _send_event({
                                    "type": "gizmo",
                                    "path": str(selected_frame.get("path") or ""),
                                    "gizmo": _selection_gizmo(
                                        camera,
                                        selected_frame,
                                        config,
                                        transform_overrides,
                                        body_transforms=current_transforms,
                                    ),
                                })
                        del products
                        remaining = frame_interval - (time.monotonic() - started)
                        if remaining > 0:
                            time.sleep(min(remaining, 0.05))
                finally:
                    failing = sys.exc_info()[0] is not None
                    for query in (
                        camera_query,
                        gizmo_query,
                        body_query,
                        bound_shape_query,
                        ghost_query,
                    ):
                        if query is not None:
                            try:
                                stage.release_query(query).wait()
                            except Exception:
                                if not failing:
                                    raise
                    for path_list in (
                        camera_paths,
                        gizmo_paths,
                        body_paths,
                        bound_shape_paths,
                        ghost_paths,
                    ):
                        if path_list is not None:
                            try:
                                paths.destroy_path_list(path_list)
                            except Exception:
                                if not failing:
                                    raise
    finally:
        if renderer is not None:
            try:
                renderer.detach_ovstage()
            except Exception:
                pass
        if stage is not None:
            stage.destroy()
        if renderer is not None:
            renderer.destroy()


def main() -> int:
    global PROTOCOL_STREAM, STREAM_TO_STDOUT
    server = None
    try:
        config = _reader()
        STREAM_TO_STDOUT = bool(config.get("http_in_parent"))
        protocol_port = int(config.get("protocol_port") or 0)
        if protocol_port:
            PROTOCOL_STREAM = socket.create_connection(("127.0.0.1", protocol_port), timeout=10.0)
            PROTOCOL_STREAM.settimeout(None)
        if not STREAM_TO_STDOUT:
            server = _serve(int(config["port"]))
        _send_event({"type": "ready", "port": int(config["port"])})
        render(config)
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        with STATE.frame_ready:
            STATE.phase = "error"
            STATE.error = message
            STATE.detail = "OVRT worker failed"
            STATE.frame_ready.notify_all()
        try:
            _send_event({"type": "error", "message": message})
        except OSError:
            pass
        traceback.print_exc(file=sys.stderr)
        # Keep the diagnostic page available until the parent closes the worker.
        while server is not None and not STATE.stop:
            time.sleep(0.1)
        return 1
    finally:
        with STATE.frame_ready:
            STATE.stop = True
            STATE.frame_ready.notify_all()
        if server is not None:
            server.shutdown()
            server.server_close()
        if PROTOCOL_STREAM is not None:
            try:
                PROTOCOL_STREAM.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
