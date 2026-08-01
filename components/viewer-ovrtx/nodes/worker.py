"""OVRT render worker and embedded browser surface.

This module deliberately has no Blacknode imports.  It is launched in a child
process so the optional native renderer has an explicit lifecycle boundary.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import socket
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
        self.latest_transforms: list[list[float]] | None = None
        self.camera_actions: list[dict[str, float | str]] = []
        self.stop = False

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "error": self.error,
                "detail": self.detail,
                "frame": self.frame_number,
                "physics_frame": self.physics_frame,
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
            })


STATE = SharedState()


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blacknode Newton · OVRT</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#080b11;color:#edf3ff}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#080b11}
#viewport{position:fixed;inset:0;display:grid;place-items:center;user-select:none;cursor:grab;touch-action:none}
#viewport.dragging{cursor:grabbing}#stream{width:100%;height:100%;object-fit:contain;display:block;pointer-events:none;-webkit-user-drag:none}
#empty{position:absolute;inset:0;display:grid;place-items:center;background:radial-gradient(circle at 50% 42%,#172131,#080b11 65%)}
#card{max-width:520px;padding:24px;text-align:center}.spinner{width:34px;height:34px;margin:0 auto 15px;border:3px solid #273449;border-top-color:#76b900;border-radius:50%;animation:s 1s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}#phase{font-weight:700;font-size:15px}#detail{margin-top:8px;color:#9ba9bd;font-size:13px;line-height:1.45}
#hud{position:fixed;left:12px;top:12px;padding:8px 10px;border:1px solid #ffffff18;border-radius:8px;background:#080b11bb;backdrop-filter:blur(8px);font-size:11px;line-height:1.5;pointer-events:none}
#hud strong{color:#76b900}#help{position:fixed;right:12px;bottom:12px;padding:7px 9px;border-radius:7px;background:#080b11aa;color:#aab5c5;font-size:10px;pointer-events:none}
</style></head><body>
<div id="viewport"><img id="stream" src="/stream.mjpg" alt="OVRT render stream" draggable="false"><div id="empty"><div id="card"><div class="spinner"></div><div id="phase">Starting NVIDIA OVRT</div><div id="detail">The first render initializes and caches RTX shaders.</div></div></div></div>
<div id="hud"><strong>OVRT</strong> · <span id="state">starting</span><br>render <span id="frame">0</span> · physics <span id="physics">0</span></div>
<div id="help">Orbit: left drag · Pan: right/middle drag · Zoom: wheel · Reset: double-click</div>
<script>
const viewport=document.querySelector('#viewport'), empty=document.querySelector('#empty'); let drag=null,pending=null,raf=0;
async function action(body){try{await fetch('/api/camera',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})}catch{}}
function flush(){raf=0;if(!pending)return;const body=pending;pending=null;action(body)}
function queueMove(kind,dx,dy){if(pending&&pending.action===kind){pending.dx+=dx;pending.dy+=dy}else pending={action:kind,dx,dy};if(!raf)raf=requestAnimationFrame(flush)}
function endDrag(e){drag=null;viewport.classList.remove('dragging');try{viewport.releasePointerCapture(e.pointerId)}catch{}}
viewport.addEventListener('pointerdown',e=>{e.preventDefault();drag={x:e.clientX,y:e.clientY,button:e.button};viewport.classList.add('dragging');viewport.setPointerCapture(e.pointerId)});
viewport.addEventListener('pointermove',e=>{if(!drag)return;e.preventDefault();const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;queueMove(drag.button===0?'orbit':'pan',dx,dy)});
viewport.addEventListener('pointerup',endDrag);
viewport.addEventListener('pointercancel',endDrag);
viewport.addEventListener('lostpointercapture',()=>{drag=null;viewport.classList.remove('dragging')});
viewport.addEventListener('dragstart',e=>e.preventDefault());
viewport.addEventListener('contextmenu',e=>e.preventDefault());
viewport.addEventListener('wheel',e=>{e.preventDefault();action({action:'zoom',delta:e.deltaY})},{passive:false});
viewport.addEventListener('dblclick',()=>action({action:'reset'}));
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
        if urlparse(self.path).path != "/api/camera":
            self._headers(HTTPStatus.NOT_FOUND, "text/plain", 0)
            return
        try:
            length = min(16384, int(self.headers.get("Content-Length") or 0))
            value = json.loads(self.rfile.read(length) or b"{}")
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
        clean = "111827"
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


def _build_wrapper_usda(config: dict[str, Any], camera_matrix: list[float]) -> str:
    source = str(config.get("asset_path") or "")
    sublayers = f"subLayers = [{_asset_reference(source)}]" if source else ""
    background = _hex_color(str(config.get("background_color") or "#111827"))
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
            body_defs.append(f'''\n        def Cube "{name}"\n        {{\n            double size = 1\n            color3f[] primvars:displayColor = [({color[0]:.6g}, {color[1]:.6g}, {color[2]:.6g})]\n            matrix4d xformOp:transform = {_matrix_text([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])}\n            uniform token[] xformOpOrder = ["xformOp:transform"]\n        }}\n''')
    grid = (
        _grid_usda(
            ground_height,
            float(config.get("grid_extent") or 1.0 / meters_per_unit),
            up_axis,
            meters_per_unit,
        )
        if bool(config.get("show_grid", True))
        else ""
    )
    ground = ""
    if bool(config.get("ground_enabled", True)):
        horizontal = 20.0 / meters_per_unit
        thickness = 0.02 / meters_per_unit
        if up_axis == "y":
            ground_matrix = [horizontal,0,0,0, 0,thickness,0,0, 0,0,horizontal,0, 0,ground_height - thickness / 2.0,0,1]
        else:
            ground_matrix = [horizontal,0,0,0, 0,horizontal,0,0, 0,0,thickness,0, 0,0,ground_height - thickness / 2.0,1]
        ground = f'''\n    def Cube "Ground"\n    {{\n        double size = 1\n        color3f[] primvars:displayColor = [(0.18, 0.2, 0.24)]\n        matrix4d xformOp:transform = {_matrix_text(ground_matrix)}\n        uniform token[] xformOpOrder = ["xformOp:transform"]\n    }}\n'''
    return f'''#usda 1.0\n(\n    {sublayers}\n    upAxis = "Z"\n    metersPerUnit = 1\n)\n\ndef Xform "BlacknodeOVRT"\n{{\n    def Camera "Camera" (\n        prepend apiSchemas = ["OmniSensorGenericCameraCoreAPI"]\n    )\n    {{\n        float focalLength = 32\n        float horizontalAperture = 36\n        float2 clippingRange = (0.01, 1000)\n        matrix4d xformOp:transform = {_matrix_text(camera_matrix)}\n        uniform token[] xformOpOrder = ["xformOp:transform"]\n    }}\n\n    def DistantLight "Key"\n    {{\n        float intensity = 2500\n        float angle = 4\n        color3f color = (1, 0.95, 0.88)\n        float xformOp:rotateXYZ = (-35, 25, -25)\n        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n    }}\n\n    def DomeLight "Sky"\n    {{\n        float intensity = 450\n        color3f color = ({background[0]:.6g}, {background[1]:.6g}, {background[2]:.6g})\n    }}\n{ground}{grid}\n    def Scope "RigidBodies"\n    {{{''.join(body_defs)}\n    }}\n\n    def Scope "Render"\n    {{\n        def RenderProduct "Viewport"\n        {{\n            int2 resolution = ({width}, {height})\n            rel camera = </BlacknodeOVRT/Camera>\n            rel orderedVars = [<LdrColor>, <HdrColor>]\n\n            def RenderVar "LdrColor"\n            {{\n                string sourceName = "LdrColor"\n            }}\n\n            def RenderVar "HdrColor"\n            {{\n                string sourceName = "HdrColor"\n            }}\n        }}\n    }}\n}}\n'''


def _wrapper_usda(config: dict[str, Any], camera_matrix: list[float]) -> str:
    """Build one composed root layer while preserving source stage semantics."""
    text = _build_wrapper_usda(config, camera_matrix)
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


def _matrix_array(matrices: list[list[float]]) -> Any:
    """Build an OVStage-compatible DLPack matrix payload."""
    import numpy as np

    return np.asarray(matrices, dtype=np.float64).reshape(-1)


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


def _encode_frame(products: Any, quality: int) -> bytes:
    for product in products.values():
        for frame in product.frames:
            import numpy as np
            import ovrtx
            from PIL import Image

            var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            try:
                view = np.from_dlpack(var)
                pixels = view.copy()
                del view
            finally:
                var.unmap()
                del var
            output = io.BytesIO()
            Image.fromarray(pixels).convert("RGB").save(output, format="JPEG", quality=quality, optimize=False)
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
    ordinal = 1
    try:
        _emit_status("initializing", "Creating NVIDIA OVRT renderer; first launch compiles RTX shaders")
        renderer = ovrtx.Renderer()
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

            with ovstage.PathDictionary(stage) as paths:
                xform = paths.intern_token("omni:xform")
                camera_paths = paths.create_path_list_from_strings(["/BlacknodeOVRT/Camera"])
                camera_query = stage.query_from_path_list(camera_paths)
                body_paths = None
                body_query = None
                if body_entries:
                    body_paths = paths.create_path_list_from_strings([str(entry["path"]) for entry in body_entries])
                    body_query = stage.query_from_path_list(body_paths)
                try:
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
                        jpeg = _encode_frame(products, int(config["jpeg_quality"]))
                        del products
                        if jpeg:
                            STATE.publish(jpeg)
                            ready = True
                            break
                    if not ready:
                        raise RuntimeError(
                            f"OVRT produced no LdrColor frame after {warmup} initialization passes"
                        )

                    frame_interval = 1.0 / float(config.get("render_fps") or 24)
                    while not STATE.stop:
                        started = time.monotonic()
                        with STATE.lock:
                            transforms = STATE.latest_transforms
                            STATE.latest_transforms = None
                            actions = STATE.camera_actions[:]
                            STATE.camera_actions.clear()
                        camera_dirty = camera.apply(actions)
                        body_matrices: list[list[float]] = []
                        if transforms is not None:
                            for entry in body_entries:
                                index = int(entry["index"])
                                if index < len(transforms):
                                    body_matrices.append(_pose_matrix(
                                        transforms[index],
                                        list(entry["scale"]),
                                        float(config.get("meters_per_unit") or 1.0),
                                        bool(entry.get("scale_in_meters", False)),
                                    ))
                        if camera_dirty or body_matrices:
                            ordinal += 1
                            if camera_dirty:
                                _write_matrices(stage, camera_query, xform, ordinal, [camera.matrix()])
                            if body_matrices and len(body_matrices) == len(body_entries):
                                _write_matrices(stage, body_query, xform, ordinal, body_matrices)
                            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                        products = renderer.step(
                            render_products={RENDER_PRODUCT}, delta_time=frame_interval, ordinal=ordinal
                        )
                        jpeg = _encode_frame(products, int(config["jpeg_quality"]))
                        if jpeg:
                            STATE.publish(jpeg)
                        del products
                        remaining = frame_interval - (time.monotonic() - started)
                        if remaining > 0:
                            time.sleep(min(remaining, 0.05))
                finally:
                    failing = sys.exc_info()[0] is not None
                    for query in (camera_query, body_query):
                        if query is not None:
                            try:
                                stage.release_query(query).wait()
                            except Exception:
                                if not failing:
                                    raise
                    for path_list in (camera_paths, body_paths):
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
