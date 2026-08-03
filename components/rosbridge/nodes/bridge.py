"""ROS 2 JointState transport for a managed Newton session via rosbridge."""
from __future__ import annotations

import math
import threading
import time
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Int, Text, node
from blacknode.pkg.blacknode_newton import runtime

try:
    import roslibpy
except Exception as exc:  # pragma: no cover - component health reports this
    roslibpy = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _IMPORT_ERROR = ""

_CATEGORY = "Newton Simulation"
_LOCK = threading.RLock()
_BRIDGES: dict[str, "RosbridgeSession"] = {}
_JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
_DIRECTIONS = {"bidirectional", "ros_to_newton", "newton_to_ros"}


def _bridge_direction(value: Any) -> str:
    direction = str(value or "bidirectional").strip().lower()
    if direction not in _DIRECTIONS:
        raise ValueError(f"unsupported Newton ROS bridge direction: {direction}")
    return direction


def _mapped_joint_positions(
    message: dict[str, Any],
    joint_map: dict[str, str],
    allowed_joints: set[str],
) -> dict[str, float]:
    names = list(message.get("name") or [])
    values = list(message.get("position") or [])
    if len(names) != len(values) or not names:
        raise ValueError("JointState command must have matching non-empty name and position arrays")
    positions: dict[str, float] = {}
    explicit_map = bool(joint_map)
    for raw_name, raw_value in zip(names, values):
        source = str(raw_name)
        if explicit_map and source not in joint_map:
            continue
        destination = joint_map.get(source, source)
        if destination not in allowed_joints:
            continue
        if destination in positions:
            raise ValueError(f"multiple ROS joints map to Newton joint '{destination}'")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"JointState command contains a non-finite position for '{source}'")
        positions[destination] = value
    if not positions:
        raise ValueError("JointState command contains no joints mapped to this Newton articulation")
    return positions


def _message_stamp_seconds(message: dict[str, Any]) -> float | None:
    """Return a ROS JointState header stamp as Unix seconds when one is present."""
    stamp = dict(dict(message.get("header") or {}).get("stamp") or {})
    if not stamp:
        return None
    seconds = float(stamp.get("sec", stamp.get("secs", 0.0)) or 0.0)
    nanoseconds = float(
        stamp.get("nanosec", stamp.get("nsec", stamp.get("nsecs", 0.0))) or 0.0
    )
    observed_at = seconds + nanoseconds / 1_000_000_000.0
    return observed_at if math.isfinite(observed_at) and observed_at > 0.0 else None


class RosbridgeSession:
    def __init__(
        self,
        bridge_id: str,
        run_id: str,
        host: str,
        port: int,
        state_topic: str,
        command_topic: str,
        publish_hz: float,
        timeout: float,
        command_stale_seconds: float,
        direction: str,
        joint_map: dict[str, Any],
    ) -> None:
        if roslibpy is None:
            raise RuntimeError(f"roslibpy is unavailable: {_IMPORT_ERROR}")
        session = runtime.get_session(run_id)
        if session is None or not session.status().get("running"):
            raise RuntimeError(f"Newton session '{run_id}' must be running first")
        try:
            from blacknode.pkg.blacknode_ros2 import rosbridge_runtime
        except Exception as exc:
            raise RuntimeError("enable blacknode-ros2/rosbridge before starting NewtonROSBridge") from exc
        self.bridge_id = bridge_id
        self.run_id = run_id
        self.host = host
        self.port = int(port)
        self.state_topic_name = state_topic
        self.command_topic_name = command_topic
        self.publish_hz = max(1.0, min(100.0, float(publish_hz)))
        self.command_stale_seconds = max(0.1, min(10.0, float(command_stale_seconds)))
        self.direction = _bridge_direction(direction)
        self.receives_commands = self.direction in {"bidirectional", "ros_to_newton"}
        self.publishes_state = self.direction in {"bidirectional", "newton_to_ros"}
        if (
            self.direction == "bidirectional"
            and state_topic.strip().rstrip("/") == command_topic.strip().rstrip("/")
        ):
            raise ValueError("bidirectional Newton ROS bridge requires separate state and command topics")
        self.allowed_joints = set(session.status().get("joint_names") or [])
        self.joint_map = {
            str(source): str(destination)
            for source, destination in dict(joint_map or {}).items()
            if str(source) and str(destination)
        }
        unknown_destinations = sorted(set(self.joint_map.values()) - self.allowed_joints)
        if unknown_destinations:
            raise ValueError(
                "joint_map contains unknown Newton joint(s): " + ", ".join(unknown_destinations)
            )
        self.ros = rosbridge_runtime.get_connection(host, self.port, timeout)
        self.state_topic = roslibpy.Topic(self.ros, state_topic, _JOINT_STATE_TYPE)
        self.command_topic = roslibpy.Topic(self.ros, command_topic, _JOINT_STATE_TYPE)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.published = 0
        self.received = 0
        self.rejected = 0
        self.last_error = ""
        self.last_command_at = 0.0
        self.stale_disarms = 0

    def start(self) -> dict[str, Any]:
        if self.publishes_state:
            self.state_topic.advertise()
        if self.receives_commands:
            self.command_topic.subscribe(self._on_command)
        self.thread = threading.Thread(
            target=self._publish_loop, daemon=True, name=f"blacknode-newton-rosbridge-{self.bridge_id}"
        )
        self.thread.start()
        return self.status()

    def _on_command(self, message: dict[str, Any]) -> None:
        try:
            positions = _mapped_joint_positions(message, self.joint_map, self.allowed_joints)
            runtime.record_joint_observation(
                self.run_id,
                positions,
                source=f"rosbridge:{self.bridge_id}",
                observed_at=_message_stamp_seconds(message),
                stale_after_seconds=self.command_stale_seconds,
            )
            self.received += 1
            runtime.command_session(
                self.run_id, positions, source=f"rosbridge:{self.bridge_id}"
            )
            self.last_command_at = time.time()
            self.last_error = ""
        except Exception as exc:  # rejected commands never terminate state publication
            self.rejected += 1
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _publish_loop(self) -> None:
        period = 1.0 / self.publish_hz
        while not self.stop_event.wait(period):
            try:
                status = runtime.session_status(self.run_id)
                if not status.get("running"):
                    self.last_error = "Newton session stopped"
                    break
                if (
                    self.receives_commands
                    and
                    status.get("armed")
                    and str(status.get("last_command_source") or "").startswith("rosbridge")
                    and self.last_command_at > 0.0
                    and time.time() - self.last_command_at > self.command_stale_seconds
                ):
                    runtime.control_session(self.run_id, "disarm")
                    self.stale_disarms += 1
                    self.last_error = (
                        f"ROS command stream stale for more than {self.command_stale_seconds:.3f}s; "
                        "simulation motion disarmed"
                    )
                    status = runtime.session_status(self.run_id)
                if self.publishes_state:
                    positions = dict(status.get("positions") or status.get("positions_radians") or {})
                    joint_names = list(status.get("joint_names") or positions)
                    now = time.time()
                    self.state_topic.publish(roslibpy.Message({
                        "header": {
                            "stamp": {"sec": int(now), "nanosec": int((now % 1.0) * 1_000_000_000)},
                            "frame_id": "newton_world",
                        },
                        "name": joint_names,
                        "position": [float(positions[name]) for name in joint_names],
                        "velocity": [],
                        "effort": [],
                    }))
                    self.published += 1
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if not self.ros.is_connected:
                    break

    def status(self) -> dict[str, Any]:
        digital_twin = dict(runtime.session_status(self.run_id).get("digital_twin") or {})
        return {
            "kind": "blacknode.newton-rosbridge", "schema_version": 1,
            "bridge_id": self.bridge_id, "run_id": self.run_id,
            "running": bool(self.thread and self.thread.is_alive()) and not self.stop_event.is_set(),
            "connected": bool(self.ros.is_connected),
            "url": f"ws://{self.host}:{self.port}",
            "direction": self.direction,
            "state_topic": self.state_topic_name, "command_topic": self.command_topic_name,
            "output_topic": self.state_topic_name if self.publishes_state else "",
            "input_topic": self.command_topic_name if self.receives_commands else "",
            "joint_map": dict(self.joint_map),
            "published": self.published, "received": self.received, "rejected": self.rejected,
            "command_stale_seconds": self.command_stale_seconds,
            "last_command_at": self.last_command_at, "stale_disarms": self.stale_disarms,
            "digital_twin": digital_twin,
            "last_error": self.last_error,
            "elapsed_seconds": max(0.0, time.time() - self.started_at),
        }

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        try:
            if self.receives_commands:
                self.command_topic.unsubscribe()
        except Exception:
            pass
        try:
            if self.publishes_state:
                self.state_topic.unadvertise()
        except Exception:
            pass
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        return self.status()


def _status(bridge_id: str) -> dict[str, Any]:
    with _LOCK:
        bridge = _BRIDGES.get(bridge_id)
    if bridge is None:
        return {
            "kind": "blacknode.newton-rosbridge", "schema_version": 1,
            "bridge_id": bridge_id, "running": False, "connected": False,
            "url": "", "direction": "", "state_topic": "", "command_topic": "",
            "output_topic": "", "input_topic": "", "joint_map": {},
            "published": 0, "received": 0, "rejected": 0, "last_error": "",
        }
    return bridge.status()


@node(
    name="NewtonROSBridge",
    component="rosbridge",
    live=True,
    category=_CATEGORY,
    description="Synchronize a managed Newton articulation with ROS 2 JointState streams over rosbridge. Direction, joint mapping, and simulation arming are explicit.",
    inputs={
        "trigger": AnyPort,
        "session": Dict(default={}),
        "action": Enum(["status", "start", "stop"], default="status"),
        "bridge_id": Text(default="so101-newton-rosbridge"),
        "run_id": Text(default="newton-scene"),
        "host": Text(default="127.0.0.1"),
        "port": Int(default=9090),
        "state_topic": Text(default="/blacknode/newton/joint_states"),
        "command_topic": Text(default="/blacknode/newton/joint_commands"),
        "direction": Enum(["bidirectional", "ros_to_newton", "newton_to_ros"], default="bidirectional"),
        "joint_map": Dict(default={}),
        "publish_hz": Float(default=30.0),
        "timeout": Float(default=10.0),
        "command_stale_seconds": Float(default=0.5),
    },
    outputs={"ok": Bool, "running": Bool, "connected": Bool, "bridge": Dict, "report": Text},
    primary_inputs=["trigger", "session"],
    primary_outputs=["bridge", "report"],
)
def newton_rosbridge(ctx: dict) -> dict:
    bridge_id = str(ctx.get("bridge_id") or "so101-newton-rosbridge").strip()
    action = str(ctx.get("action") or "status").lower()
    session = dict(ctx.get("session") or {})
    run_id = str(session.get("run_id") or ctx.get("run_id") or "newton-scene")
    try:
        if action == "start":
            with _LOCK:
                prior = _BRIDGES.get(bridge_id)
                if prior is not None and prior.status().get("running"):
                    status = prior.status()
                else:
                    bridge = RosbridgeSession(
                        bridge_id,
                        run_id,
                        str(ctx.get("host") or "127.0.0.1"),
                        int(ctx.get("port") or 9090),
                        str(ctx.get("state_topic") or "/blacknode/newton/joint_states"),
                        str(ctx.get("command_topic") or "/blacknode/newton/joint_commands"),
                        float(ctx.get("publish_hz") or 30.0),
                        float(ctx.get("timeout") or 10.0),
                        float(ctx.get("command_stale_seconds") or 0.5),
                        str(ctx.get("direction") or "bidirectional"),
                        dict(ctx.get("joint_map") or {}),
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
        return {
            "ok": ok, "running": bool(status.get("running")),
            "connected": bool(status.get("connected")), "bridge": status,
            "report": (
                f"Newton ROS bridge {'connected' if status.get('connected') else 'stopped'}"
                f" at {status.get('url') or 'ws://not-started'}"
                + (f"; {status['last_error']}" if status.get("last_error") else "")
            ),
        }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        failed = {
            **_status(bridge_id),
            "run_id": run_id,
            "direction": str(ctx.get("direction") or "bidirectional"),
            "state_topic": str(ctx.get("state_topic") or "/blacknode/newton/joint_states"),
            "command_topic": str(ctx.get("command_topic") or "/blacknode/newton/joint_commands"),
            "joint_map": dict(ctx.get("joint_map") or {}),
            "last_error": error,
        }
        return {
            "ok": False, "running": False, "connected": False, "bridge": failed,
            "report": f"Newton ROS bridge FAILED: {error}",
        }


def stop_rosbridge_services() -> dict[str, Any]:
    with _LOCK:
        bridges = list(_BRIDGES.values())
        _BRIDGES.clear()
    for bridge in bridges:
        bridge.stop()
    return {"ok": True, "stopped": len(bridges)}


runtime.register_shutdown_hook("rosbridge", stop_rosbridge_services)
