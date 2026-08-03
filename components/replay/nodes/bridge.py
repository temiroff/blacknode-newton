"""Dataset episode replay transport for a managed Newton articulation."""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any
from urllib.parse import urlparse

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Text, node
from blacknode.pkg.blacknode_newton import runtime

try:
    import websocket
except Exception as exc:  # pragma: no cover - component health reports this
    websocket = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _IMPORT_ERROR = ""

_CATEGORY = "Newton Simulation"
_LOCK = threading.RLock()
_BRIDGES: dict[str, "ReplayBridgeSession"] = {}


def _stream_url(value: Any) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("stream_url must be a ws:// or wss:// Dataset StreamPublisher URL")
    return url


def _mapped_replay_positions(
    message: dict[str, Any],
    joint_map: dict[str, str],
    allowed_joints: set[str],
    joint_units: dict[str, str],
    units: str = "auto",
) -> dict[str, float]:
    names = [str(name) for name in list(message.get("joint_names") or [])]
    values = list(message.get("positions") or [])
    if not values:
        return {}
    if len(names) != len(values):
        raise ValueError("replay frame must have matching joint_names and positions arrays")
    source_units = str((message.get("units") or "radians") if units == "auto" else units or "radians").lower()
    if source_units not in {"radians", "degrees"}:
        raise ValueError(f"unsupported replay joint units: {source_units}")
    explicit_map = bool(joint_map)
    positions: dict[str, float] = {}
    for source, raw_value in zip(names, values):
        if explicit_map and source not in joint_map:
            continue
        destination = joint_map.get(source, source)
        if destination not in allowed_joints:
            continue
        if destination in positions:
            raise ValueError(f"multiple dataset joints map to Newton joint '{destination}'")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"replay frame contains a non-finite position for '{source}'")
        if source_units == "degrees" and joint_units.get(destination, "radians") == "radians":
            value = math.radians(value)
        positions[destination] = value
    if not positions:
        raise ValueError("replay frame contains no joints mapped to this Newton articulation")
    return positions


class ReplayBridgeSession:
    def __init__(
        self,
        bridge_id: str,
        run_id: str,
        stream_url: str,
        joint_map: dict[str, Any],
        units: str,
        apply_seek: bool,
        timeout: float,
        command_stale_seconds: float,
    ) -> None:
        if websocket is None:
            raise RuntimeError(f"websocket-client is unavailable: {_IMPORT_ERROR}")
        session = runtime.get_session(run_id)
        if session is None or not session.status().get("running"):
            raise RuntimeError(f"Newton session '{run_id}' must be running first")
        self.bridge_id = str(bridge_id)
        self.run_id = str(run_id)
        self.stream_url = _stream_url(stream_url)
        self.timeout = max(0.1, min(30.0, float(timeout)))
        self.command_stale_seconds = max(0.1, min(10.0, float(command_stale_seconds)))
        self.units = str(units or "auto").lower()
        if self.units not in {"auto", "radians", "degrees"}:
            raise ValueError(f"unsupported replay units setting: {self.units}")
        self.apply_seek = bool(apply_seek)
        self.allowed_joints = set(session.status().get("joint_names") or [])
        self.joint_units = dict(getattr(session, "joint_units", {}) or {})
        self.joint_map = {
            str(source): str(destination)
            for source, destination in dict(joint_map or {}).items()
            if str(source) and str(destination)
        }
        unknown = sorted(set(self.joint_map.values()) - self.allowed_joints)
        if unknown:
            raise ValueError("joint_map contains unknown Newton joint(s): " + ", ".join(unknown))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.socket: Any = None
        self.started_at = time.time()
        self.connected = False
        self.received = 0
        self.applied = 0
        self.rejected = 0
        self.seek_frames = 0
        self.last_frame_index = -1
        self.last_command_at = 0.0
        self.stale_disarms = 0
        self.last_error = ""

    def start(self) -> dict[str, Any]:
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"blacknode-newton-replay-{self.bridge_id}",
        )
        self.thread.start()
        return self.status()

    def _check_stale(self) -> None:
        if self.last_command_at <= 0.0:
            return
        status = runtime.session_status(self.run_id)
        if (
            status.get("armed")
            and status.get("last_command_source") == "dataset-replay"
            and time.time() - self.last_command_at > self.command_stale_seconds
        ):
            self._disarm_owned_command(
                f"dataset replay stale for more than {self.command_stale_seconds:.3f}s; "
                "simulation motion disarmed"
            )

    def _disarm_owned_command(self, reason: str) -> None:
        status = runtime.session_status(self.run_id)
        if status.get("armed") and status.get("last_command_source") == "dataset-replay":
            runtime.control_session(self.run_id, "disarm")
            self.stale_disarms += 1
            self.last_command_at = 0.0
            self.last_error = reason

    def _on_message(self, message: dict[str, Any]) -> None:
        if message.get("kind") == "blacknode.stream-schema" or not message.get("positions"):
            return
        self.received += 1
        playback_event = str(message.get("playback_event") or "autoplay")
        if playback_event == "seek" and not self.apply_seek:
            return
        try:
            positions = _mapped_replay_positions(
                message,
                self.joint_map,
                self.allowed_joints,
                self.joint_units,
                self.units,
            )
            runtime.record_joint_observation(
                self.run_id,
                positions,
                source=f"dataset-replay:{self.bridge_id}",
                stale_after_seconds=self.command_stale_seconds,
            )
            runtime.command_session(self.run_id, positions, source="dataset-replay")
            self.applied += 1
            self.last_frame_index = int(message.get("frame_index") or 0)
            self.last_command_at = time.time()
            if playback_event == "seek":
                self.seek_frames += 1
            self.last_error = ""
        except Exception as exc:
            self.rejected += 1
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _loop(self) -> None:
        try:
            self.socket = websocket.create_connection(
                self.stream_url,
                timeout=self.timeout,
                enable_multithread=True,
            )
            self.socket.settimeout(0.25)
            self.connected = True
            while not self.stop_event.is_set():
                try:
                    payload = self.socket.recv()
                except websocket.WebSocketTimeoutException:
                    self._check_stale()
                    continue
                if payload is None or payload == "":
                    break
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError("replay stream message must be a JSON object")
                self._on_message(value)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.connected = False
            try:
                if self.socket is not None:
                    self.socket.close()
            except Exception:
                pass
            try:
                if self.last_command_at > 0.0:
                    self._disarm_owned_command(
                        "dataset replay disconnected; simulation motion disarmed"
                    )
            except Exception as exc:
                if not self.last_error:
                    self.last_error = f"failed to disarm disconnected replay: {type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "kind": "blacknode.newton-replay-bridge",
            "schema_version": 1,
            "bridge_id": self.bridge_id,
            "run_id": self.run_id,
            "running": bool(self.thread and self.thread.is_alive()) and not self.stop_event.is_set(),
            "connected": bool(self.connected),
            "stream_url": self.stream_url,
            "joint_map": dict(self.joint_map),
            "units": self.units,
            "apply_seek": self.apply_seek,
            "received": self.received,
            "applied": self.applied,
            "rejected": self.rejected,
            "seek_frames": self.seek_frames,
            "last_frame_index": self.last_frame_index,
            "last_command_at": self.last_command_at,
            "command_stale_seconds": self.command_stale_seconds,
            "stale_disarms": self.stale_disarms,
            "last_error": self.last_error,
            "elapsed_seconds": max(0.0, time.time() - self.started_at),
        }

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        try:
            if self.socket is not None:
                self.socket.close()
        except Exception:
            pass
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        return self.status()


def _status(bridge_id: str) -> dict[str, Any]:
    with _LOCK:
        bridge = _BRIDGES.get(bridge_id)
    if bridge is not None:
        return bridge.status()
    return {
        "kind": "blacknode.newton-replay-bridge",
        "schema_version": 1,
        "bridge_id": bridge_id,
        "run_id": "",
        "running": False,
        "connected": False,
        "stream_url": "",
        "joint_map": {},
        "received": 0,
        "applied": 0,
        "rejected": 0,
        "last_error": "",
    }


@node(
    name="NewtonReplayBridge",
    component="replay",
    live=True,
    category=_CATEGORY,
    description=(
        "Drive a managed Newton articulation from Dataset Browser episode playback through a "
        "StreamPublisher WebSocket. Joint mapping, units, seeks, and stale-data disarm are explicit; "
        "the Newton simulation must be armed separately."
    ),
    inputs={
        "trigger": AnyPort,
        "session": Dict(default={}),
        "action": Enum(["status", "start", "stop"], default="status"),
        "bridge_id": Text(default="dataset-newton-replay"),
        "run_id": Text(default=runtime.WORKSPACE_RUN_ID),
        "stream_url": Text(default="ws://127.0.0.1:8765"),
        "joint_map": Dict(default={}),
        "units": Enum(["auto", "radians", "degrees"], default="auto"),
        "apply_seek": Bool(default=True),
        "timeout": Float(default=5.0),
        "command_stale_seconds": Float(default=0.5),
    },
    outputs={"ok": Bool, "running": Bool, "connected": Bool, "bridge": Dict, "report": Text},
    primary_inputs=["trigger", "session", "stream_url"],
    primary_outputs=["bridge", "report"],
)
def newton_replay_bridge(ctx: dict) -> dict:
    bridge_id = str(ctx.get("bridge_id") or "dataset-newton-replay").strip()
    action = str(ctx.get("action") or "status").lower()
    session = dict(ctx.get("session") or {})
    run_id = str(session.get("run_id") or ctx.get("run_id") or runtime.WORKSPACE_RUN_ID)
    try:
        if action == "start":
            with _LOCK:
                prior = _BRIDGES.get(bridge_id)
                if prior is not None and prior.status().get("running"):
                    status = prior.status()
                else:
                    bridge = ReplayBridgeSession(
                        bridge_id,
                        run_id,
                        str(ctx.get("stream_url") or "ws://127.0.0.1:8765"),
                        dict(ctx.get("joint_map") or {}),
                        str(ctx.get("units") or "auto"),
                        bool(ctx.get("apply_seek", True)),
                        float(ctx.get("timeout") or 5.0),
                        float(ctx.get("command_stale_seconds") or 0.5),
                    )
                    _BRIDGES[bridge_id] = bridge
                    status = bridge.start()
        elif action == "stop":
            with _LOCK:
                bridge = _BRIDGES.pop(bridge_id, None)
            status = bridge.stop() if bridge is not None else _status(bridge_id)
        else:
            status = _status(bridge_id)
        ok = not bool(status.get("last_error"))
        state = "connected" if status.get("connected") else "starting" if status.get("running") else "stopped"
        return {
            "ok": ok,
            "running": bool(status.get("running")),
            "connected": bool(status.get("connected")),
            "bridge": status,
            "report": (
                f"Newton dataset replay {state}"
                f" at {status.get('stream_url') or 'ws://not-started'}"
                + (f"; {status['last_error']}" if status.get("last_error") else "")
            ),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failed = {
            **_status(bridge_id),
            "run_id": run_id,
            "stream_url": str(ctx.get("stream_url") or ""),
            "joint_map": dict(ctx.get("joint_map") or {}),
            "last_error": error,
        }
        return {
            "ok": False,
            "running": False,
            "connected": False,
            "bridge": failed,
            "report": f"Newton dataset replay FAILED: {error}",
        }


def stop_replay_services() -> dict[str, Any]:
    with _LOCK:
        bridges = list(_BRIDGES.values())
        _BRIDGES.clear()
    for bridge in bridges:
        bridge.stop()
    return {"ok": True, "stopped": len(bridges)}


runtime.register_shutdown_hook("dataset-replay", stop_replay_services)
