"""Isolated NVIDIA OVRT implementation of the Newton viewer contract."""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from blacknode.pkg.blacknode_newton.viewer_contract import register_viewer


_VIEWER_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blacknode Newton · OVRT</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#080b11;color:#edf3ff}*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#080b11}#v{position:fixed;inset:0;display:grid;place-items:center;cursor:grab;user-select:none;touch-action:none}#v.drag{cursor:grabbing}#s{width:100%;height:100%;object-fit:contain;pointer-events:none;-webkit-user-drag:none}#e{position:absolute;inset:0;display:grid;place-items:center;background:radial-gradient(circle at 50% 42%,#172131,#080b11 65%)}#c{max-width:520px;padding:24px;text-align:center}.spin{width:34px;height:34px;margin:0 auto 15px;border:3px solid #273449;border-top-color:#76b900;border-radius:50%;animation:r 1s linear infinite}@keyframes r{to{transform:rotate(360deg)}}#s1{font-weight:700;font-size:15px}#d{margin-top:8px;color:#9ba9bd;font-size:13px;line-height:1.45}#h{position:fixed;left:12px;top:12px;padding:8px 10px;border:1px solid #ffffff18;border-radius:8px;background:#080b11bb;font-size:11px;line-height:1.5;pointer-events:none}#h strong{color:#76b900}#help{position:fixed;right:12px;bottom:12px;padding:7px 9px;border-radius:7px;background:#080b11aa;color:#aab5c5;font-size:10px;pointer-events:none}
</style></head><body><div id="v"><img id="s" src="/stream.mjpg" draggable="false"><div id="e"><div id="c"><div class="spin"></div><div id="s1">Starting NVIDIA OVRT</div><div id="d">The first render initializes and caches RTX shaders.</div></div></div></div><div id="h"><strong>OVRT</strong> · <span id="p">starting</span><br>render <span id="f">0</span> · physics <span id="pf">0</span></div><div id="help">Orbit: left drag · Pan: right/middle drag · Zoom: wheel · Reset: double-click</div><script>
const v=document.querySelector('#v'),e=document.querySelector('#e');let drag=null,pending=null,raf=0;async function a(x){try{await fetch('/api/camera',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)})}catch{}}function flush(){raf=0;if(!pending)return;const x=pending;pending=null;a(x)}function move(action,dx,dy){if(pending&&pending.action===action){pending.dx+=dx;pending.dy+=dy}else pending={action,dx,dy};if(!raf)raf=requestAnimationFrame(flush)}function end(x){drag=null;v.classList.remove('drag');try{v.releasePointerCapture(x.pointerId)}catch{}}v.onpointerdown=x=>{x.preventDefault();drag={x:x.clientX,y:x.clientY,b:x.button};v.classList.add('drag');v.setPointerCapture(x.pointerId)};v.onpointermove=x=>{if(!drag)return;x.preventDefault();const dx=x.clientX-drag.x,dy=x.clientY-drag.y;drag.x=x.clientX;drag.y=x.clientY;move(drag.b===0?'orbit':'pan',dx,dy)};v.onpointerup=end;v.onpointercancel=end;v.onlostpointercapture=()=>{drag=null;v.classList.remove('drag')};v.ondragstart=x=>x.preventDefault();v.oncontextmenu=x=>x.preventDefault();v.onwheel=x=>{x.preventDefault();a({action:'zoom',delta:x.deltaY})};v.ondblclick=()=>a({action:'reset'});async function st(){try{const x=await(await fetch('/api/status',{cache:'no-store'})).json();document.querySelector('#p').textContent=x.error?'error':x.phase;document.querySelector('#f').textContent=x.frame;document.querySelector('#pf').textContent=x.physics_frame;document.querySelector('#s1').textContent=x.error?'OVRT render failed':'Starting NVIDIA OVRT';document.querySelector('#d').textContent=x.error||x.detail;if(x.has_image)e.style.display='none'}catch{document.querySelector('#p').textContent='disconnected'}}setInterval(st,750);st();
</script></body></html>"""


def _loopback_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def _select_viewer_port(requested: int) -> int:
    requested = max(1024, min(65535, int(requested)))
    if _loopback_port_available(requested):
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _source_stage_info(asset_path: str) -> dict[str, Any]:
    """Inspect render-space metadata without making OVRT depend on Newton internals."""
    info: dict[str, Any] = {
        "meters_per_unit": 1.0,
        "up_axis": "z",
        "bounds_min": [],
        "bounds_max": [],
    }
    if not asset_path:
        return info
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return info
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if math.isfinite(meters_per_unit) and meters_per_unit > 0.0:
            info["meters_per_unit"] = meters_per_unit
        up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Z").lower()
        if up_axis in {"y", "z"}:
            info["up_axis"] = up_axis
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        if not bounds.IsEmpty():
            lower = [float(value) for value in bounds.GetMin()]
            upper = [float(value) for value in bounds.GetMax()]
            if all(math.isfinite(value) for value in lower + upper):
                info["bounds_min"] = lower
                info["bounds_max"] = upper
    except Exception:
        # Scene construction already reports malformed USD. Camera inspection is
        # best-effort so an unusual boundable can still reach the renderer.
        pass
    return info


def _scene_camera(
    session: Any,
    config: dict[str, Any],
    stage_info: dict[str, Any],
) -> tuple[list[float], list[float], str]:
    camera = dict(config.get("camera") or {})
    position = [float(value) for value in list(camera.get("position_m") or [])]
    target = [float(value) for value in list(camera.get("target_m") or [])]
    up_axis = str(camera.get("up_axis") or "auto").lower()
    source_up_axis = str(stage_info.get("up_axis") or "z")
    meters_per_unit = max(1.0e-12, float(stage_info.get("meters_per_unit") or 1.0))
    if len(position) == 3 and len(target) == 3:
        return (
            [value / meters_per_unit for value in position],
            [value / meters_per_unit for value in target],
            source_up_axis if up_axis == "auto" else up_axis,
        )

    lower = list(stage_info.get("bounds_min") or [])
    upper = list(stage_info.get("bounds_max") or [])
    if len(lower) == 3 and len(upper) == 3:
        center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
        extent = max(upper[axis] - lower[axis] for axis in range(3))
        distance = max(0.8 / meters_per_unit, extent * 1.8)
        if source_up_axis == "y":
            return (
                [center[0] + distance, center[1] + distance * 0.75, center[2] + distance],
                center,
                "y",
            )
        return (
            [center[0] + distance, center[1] - distance, center[2] + distance * 0.75],
            center,
            "z",
        )

    points: list[list[float]] = []
    try:
        for transform in session.state_0.body_q.numpy():
            point = [float(value) for value in transform[:3]]
            if all(math.isfinite(value) for value in point):
                points.append(point)
    except Exception:
        pass
    if not points:
        center = [0.0, 0.0, 0.15]
        extent = 0.5
    else:
        lower = [min(point[axis] for point in points) for axis in range(3)]
        upper = [max(point[axis] for point in points) for axis in range(3)]
        center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
        extent = max(0.4, max(upper[axis] - lower[axis] for axis in range(3)))
    distance = max(0.8, extent * 2.2)
    position = [center[0] + distance, center[1] - distance, center[2] + distance * 0.75]
    return (
        [value / meters_per_unit for value in position],
        [value / meters_per_unit for value in center],
        source_up_axis,
    )


def _body_entries(session: Any, model: Any) -> list[dict[str, Any]]:
    runtime_bodies = {
        str(body.get("name") or ""): body
        for body in list(session.scene.get("rigid_bodies") or [])
    }
    entries: list[dict[str, Any]] = []
    for index, raw_label in enumerate(list(model.body_label)):
        label = str(raw_label or "")
        if label.startswith("/"):
            entries.append({"index": index, "path": label, "scale": [1.0, 1.0, 1.0]})
            continue
        body = runtime_bodies.get(label)
        if body is None:
            continue
        entries.append({
            "index": index,
            "path": f"/BlacknodeOVRT/RigidBodies/body_{index}",
            "name": label,
            "scale": [float(value) for value in body.get("size_m", [0.05, 0.05, 0.05])],
            "scale_in_meters": True,
            "color": [float(value) for value in body.get("color_rgb", [0.8, 0.25, 0.12])],
        })
    return entries


def _state_transforms(state: Any) -> list[list[float]]:
    """Return Newton body poses while treating a valid empty stage as empty."""
    body_q = getattr(state, "body_q", None)
    if body_q is None:
        return []
    return body_q.numpy().tolist()


class OVRTViewer:
    """Proxy a live Newton session to an isolated OVRT/OVStage renderer."""

    def __init__(self, session: Any, model: Any, config: dict[str, Any]) -> None:
        self.session = session
        self.requested_port = int(config.get("port") or 8080)
        self.port = _select_viewer_port(self.requested_port)
        self._closed = False
        self._ready = threading.Event()
        self._last_error = ""
        self._diagnostics: list[str] = []
        self._web_lock = threading.RLock()
        self._frame_ready = threading.Condition(self._web_lock)
        self._jpeg = b""
        self._render_frame = 0
        self._physics_frame = 0
        self._web_phase = "starting"
        self._web_detail = "Starting isolated OVRT render worker"
        self._sender_started = False
        self._updates: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)

        self._http_server = self._start_http_server()
        self._protocol_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._protocol_listener.bind(("127.0.0.1", 0))
        self._protocol_listener.listen(1)
        self._protocol_listener.settimeout(15.0)
        self._protocol_socket: socket.socket | None = None

        asset_path = str(session.scene.get("asset_path") or "")
        stage_info = _source_stage_info(asset_path)
        position, target, up_axis = _scene_camera(session, config, stage_info)
        width = max(320, min(3840, int(config.get("width") or 1280)))
        height = max(240, min(2160, int(config.get("height") or 720)))
        meters_per_unit = float(stage_info["meters_per_unit"])
        bounds_min = list(stage_info.get("bounds_min") or [])
        bounds_max = list(stage_info.get("bounds_max") or [])
        bounds_extent = (
            max(bounds_max[axis] - bounds_min[axis] for axis in range(3))
            if len(bounds_min) == 3 and len(bounds_max) == 3
            else 0.0
        )
        worker_config = {
            "port": self.port,
            "label": str(config.get("label") or "Blacknode Newton · OVRT"),
            "asset_path": asset_path,
            "ground_enabled": bool(session.scene.get("ground", {}).get("enabled", True)),
            "ground_height": float(session.scene.get("ground", {}).get("height_m", 0.0)),
            "meters_per_unit": meters_per_unit,
            "up_axis": up_axis,
            "grid_extent": max(1.0 / meters_per_unit, bounds_extent * 0.6),
            "show_grid": bool(config.get("show_grid", True)),
            "background_color": str(config.get("background_color") or "#111827"),
            "body_entries": _body_entries(session, model),
            "camera": {"position": position, "target": target, "up_axis": up_axis},
            "width": width,
            "height": height,
            "render_fps": max(1, min(60, int(config.get("render_fps") or 24))),
            # OVRT may return no LdrColor while shaders are compiling. The worker
            # treats this as a maximum readiness budget and exits early on the
            # first usable frame.
            "warmup_frames": max(1, min(120, int(config.get("warmup_frames", 40)))),
            "jpeg_quality": max(50, min(98, int(config.get("jpeg_quality") or 90))),
            "http_in_parent": True,
            "protocol_port": int(self._protocol_listener.getsockname()[1]),
        }
        worker_path = Path(__file__).with_name("worker.py")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                [sys.executable, str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                creationflags=0,
            )
        except Exception:
            self._closed = True
            self._protocol_listener.close()
            self._http_server.shutdown()
            self._http_server.server_close()
            raise
        self._protocol_thread = threading.Thread(target=self._read_protocol, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._sender_thread = threading.Thread(target=self._send_updates, daemon=True)
        self._protocol_thread.start()
        self._stderr_thread.start()
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps({"type": "configure", "config": worker_config}) + "\n")
        self._process.stdin.flush()
        if not self._ready.wait(10.0):
            try:
                self._raise_if_exited()
                detail = "OVRT viewer worker did not establish its control channel within 10 seconds"
            except RuntimeError as exc:
                detail = str(exc)
            self.close()
            raise RuntimeError(detail)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _read_protocol(self) -> None:
        try:
            connection, _address = self._protocol_listener.accept()
            self._protocol_socket = connection
        except OSError as exc:
            if not self._closed:
                self._last_error = f"OVRT protocol connection failed: {exc}"
            return
        with connection, connection.makefile("r", encoding="utf-8") as stream:
            for raw_line in stream:
                self._handle_worker_event(raw_line)

    def _handle_worker_event(self, raw_line: str) -> None:
        line = raw_line.strip()
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            if line:
                self._diagnostics.append(line)
            return
        if event.get("type") == "ready":
            with self._web_lock:
                self._web_phase = "initializing"
                self._web_detail = "Creating NVIDIA OVRT renderer; first launch compiles RTX shaders"
            self._ready.set()
        elif event.get("type") == "status":
            with self._web_lock:
                self._web_phase = str(event.get("phase") or "initializing")
                self._web_detail = str(event.get("detail") or "Starting NVIDIA OVRT")
                if self._web_phase in {"loading", "warming", "streaming"} and not self._sender_started:
                    self._sender_started = True
                    self._sender_thread.start()
        elif event.get("type") == "frame":
            try:
                frame = base64.b64decode(str(event.get("jpeg") or ""), validate=True)
            except (ValueError, TypeError):
                return
            with self._frame_ready:
                self._jpeg = frame
                self._render_frame += 1
                self._web_phase = "streaming"
                self._web_detail = "OVRT RTX render stream"
                self._frame_ready.notify_all()
        elif event.get("type") == "error":
            self._last_error = str(event.get("message") or "OVRT worker failed")
            with self._frame_ready:
                self._web_phase = "error"
                self._web_detail = "OVRT worker failed"
                self._frame_ready.notify_all()

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        pending = ""
        while True:
            chunk = self._process.stderr.read(4096)
            if not chunk:
                break
            pending += chunk
            lines = pending.splitlines(keepends=True)
            pending = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                pending = lines.pop()
            for line in lines:
                clean = line.strip()
                if clean:
                    self._diagnostics.append(clean)
                    del self._diagnostics[:-50]
        if pending.strip():
            self._diagnostics.append(pending.strip())

    def _send_updates(self) -> None:
        while True:
            update = self._updates.get()
            if update is None:
                return
            try:
                if self._process.stdin is None:
                    return
                self._process.stdin.write(json.dumps(update, separators=(",", ":")) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return

    def _raise_if_exited(self) -> None:
        code = self._process.poll()
        if code is None:
            return
        detail = self._last_error or (self._diagnostics[-1] if self._diagnostics else "no diagnostics")
        raise RuntimeError(f"OVRT viewer worker exited with code {code}: {detail}")

    def _start_http_server(self) -> ThreadingHTTPServer:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send_headers(self, status: int, content_type: str, length: int | None = None) -> None:
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
                    body = _VIEWER_HTML.encode("utf-8")
                    self._send_headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
                    self.wfile.write(body)
                    return
                if path == "/api/status":
                    with viewer._web_lock:
                        status = {
                            "phase": viewer._web_phase,
                            "error": viewer._last_error,
                            "detail": viewer._web_detail,
                            "frame": viewer._render_frame,
                            "physics_frame": viewer._physics_frame,
                            "has_image": bool(viewer._jpeg),
                        }
                    body = json.dumps(status).encode("utf-8")
                    self._send_headers(HTTPStatus.OK, "application/json", len(body))
                    self.wfile.write(body)
                    return
                if path == "/stream.mjpg":
                    self._send_headers(HTTPStatus.OK, "multipart/x-mixed-replace; boundary=frame")
                    seen = -1
                    try:
                        while not viewer._closed:
                            with viewer._frame_ready:
                                viewer._frame_ready.wait_for(
                                    lambda: viewer._render_frame != seen or viewer._closed, timeout=2.0
                                )
                                if viewer._closed:
                                    return
                                seen = viewer._render_frame
                                frame = viewer._jpeg
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
                self._send_headers(HTTPStatus.NOT_FOUND, "text/plain", 0)

            def do_POST(self) -> None:  # noqa: N802
                if urlparse(self.path).path != "/api/camera":
                    self._send_headers(HTTPStatus.NOT_FOUND, "text/plain", 0)
                    return
                try:
                    length = min(16384, int(self.headers.get("Content-Length") or 0))
                    value = json.loads(self.rfile.read(length) or b"{}")
                    action = str(value.get("action") or "")
                    if action not in {"orbit", "pan", "zoom", "reset"}:
                        raise ValueError("unsupported camera action")
                    event: dict[str, Any] = {"type": "camera", "action": action}
                    for name in ("dx", "dy", "delta"):
                        if name in value:
                            event[name] = max(-500.0, min(500.0, float(value[name])))
                    viewer._queue_update(event)
                    self._send_headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                except (ValueError, TypeError, json.JSONDecodeError):
                    self._send_headers(HTTPStatus.BAD_REQUEST, "text/plain", 0)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = False

        server = Server(("127.0.0.1", self.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="ovrtx-proxy-http").start()
        return server

    def _queue_update(self, update: dict[str, Any]) -> None:
        try:
            self._updates.put_nowait(update)
        except queue.Full:
            try:
                self._updates.get_nowait()
            except queue.Empty:
                pass
            try:
                self._updates.put_nowait(update)
            except queue.Full:
                pass

    def is_running(self) -> bool:
        if self._closed:
            return False
        self._raise_if_exited()
        return True

    def begin_frame(self, time_seconds: float) -> None:
        del time_seconds

    def log_state(self, state: Any) -> None:
        self._raise_if_exited()
        transforms = _state_transforms(state)
        update = {"type": "poses", "transforms": transforms, "frame": self.session.frame_count}
        with self._web_lock:
            self._physics_frame = int(self.session.frame_count)
        self._queue_update(update)

    def end_frame(self) -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._sender_started:
            try:
                self._updates.put_nowait(None)
            except queue.Full:
                try:
                    self._updates.get_nowait()
                    self._updates.put_nowait(None)
                except queue.Empty:
                    pass
        try:
            if self._process.stdin is not None:
                self._process.stdin.write('{"type":"close"}\n')
                self._process.stdin.flush()
                self._process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        try:
            self._protocol_listener.close()
        except OSError:
            pass
        if self._protocol_socket is not None:
            try:
                self._protocol_socket.close()
            except OSError:
                pass
        self._http_server.shutdown()
        self._http_server.server_close()


def _factory(session: Any, model: Any, config: dict[str, Any]) -> OVRTViewer:
    return OVRTViewer(session, model, config)


register_viewer("ovrtx", _factory)
