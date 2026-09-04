"""Typed Blacknode nodes for Newton scene creation and teleoperation."""
from __future__ import annotations

import json
from pathlib import Path

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Int, List, Text, node

from . import runtime

runtime_status = runtime.runtime_status
stop_runtime_services = runtime.stop_runtime_services

_CATEGORY = "Newton Simulation"
_DEFAULT_ASSET = runtime.package_asset_uri("assets/so101_robot.usd")


def _xacro_arguments(value: object) -> dict[str, object]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "xacro_arguments must be a JSON object such as "
            '{"variant":"arm","tool":"gripper"}'
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("xacro_arguments must contain one JSON object")
    return parsed


@node(
    name="NewtonScene",
    component="runtime",
    category=_CATEGORY,
    description=(
        "Load a user-selected USD, URDF, Xacro, or MuJoCo scene for Newton "
        "visualization and physics. No model is selected by default."
    ),
    inputs={
        "trigger": AnyPort,
        "asset_path": Text(default=""),
        "root_path": Text(default="/"),
        "fixed_base": Bool(default=True),
        "ground_enabled": Bool(default=True),
        "ground_height": Float(default=0.0),
        "self_collisions": Bool(default=False),
        "show_colliders": Bool(default=False),
        "xacro_arguments": Text(default="{}"),
    },
    outputs={"ok": Bool, "scene": Dict, "report": Text},
    primary_inputs=["trigger", "asset_path"],
    primary_outputs=["scene", "report"],
)
def newton_scene(ctx: dict) -> dict:
    asset_path = str(ctx.get("asset_path") or "").strip()
    try:
        if not asset_path:
            raise ValueError(
                "asset_path is required; choose a USD, URDF, Xacro, XML, or MJCF file"
            )
        suffix = Path(asset_path).suffix.lower()
        common = {
            "asset_path": asset_path,
            "fixed_base": bool(ctx.get("fixed_base", True)),
            "ground_enabled": bool(ctx.get("ground_enabled", True)),
            "ground_height": float(ctx.get("ground_height") or 0.0),
            "self_collisions": bool(ctx.get("self_collisions", False)),
            "show_colliders": bool(ctx.get("show_colliders", False)),
        }
        if suffix in {".usd", ".usda", ".usdc"}:
            scene = runtime.make_usd_scene_spec(
                **common,
                root_path=str(ctx.get("root_path") or "/"),
                home_positions={},
                rigid_bodies=[],
                particle_fill={},
                convex_decomposition_patterns=[],
                friction_overrides={},
            )
        elif suffix in {".urdf", ".xacro"}:
            scene = runtime.make_robot_description_scene_spec(
                **common,
                xacro_arguments=_xacro_arguments(ctx.get("xacro_arguments")),
            )
        elif suffix in {".xml", ".mjcf"}:
            scene = runtime.make_mjcf_scene_spec(**common)
        else:
            raise ValueError(
                f"unsupported scene format {suffix or '<none>'}; "
                "choose USD, URDF, Xacro, XML, or MJCF"
            )
        scene_format = str(
            scene.get("robot_description_format") or suffix.lstrip(".") or "scene"
        ).upper()
        return {
            "ok": True,
            "scene": scene,
            "report": (
                f"Newton {scene_format} scene ready: "
                f"{scene['asset_path']}; simulation motion remains disarmed"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "scene": {},
            "report": f"Newton scene FAILED: {type(exc).__name__}: {exc}",
        }


@node(
    name="NewtonUSDScene",
    component="runtime",
    category=_CATEGORY,
    description="Load a USD scene for Newton physics with inspected articulation joints, optional rigid bodies, and a procedural particle fill.",
    inputs={
        "trigger": AnyPort,
        "asset_path": Text(default=_DEFAULT_ASSET),
        "root_path": Text(default="/"),
        "fixed_base": Bool(default=True),
        "ground_enabled": Bool(default=True),
        "ground_height": Float(default=0.0),
        "self_collisions": Bool(default=False),
        "show_colliders": Bool(default=False),
        "home_positions": Dict(default={}),
        "rigid_bodies": List(default=[]),
        "particle_fill": Dict(default={}),
        "convex_decomposition_patterns": List(default=[]),
        "friction_overrides": Dict(default={}),
    },
    outputs={"ok": Bool, "scene": Dict, "report": Text},
    primary_inputs=["trigger", "asset_path"],
    primary_outputs=["scene", "report"],
)
def newton_usd_scene(ctx: dict) -> dict:
    try:
        scene = runtime.make_usd_scene_spec(
            asset_path=str(ctx.get("asset_path") or _DEFAULT_ASSET),
            root_path=str(ctx.get("root_path") or "/"),
            fixed_base=bool(ctx.get("fixed_base", True)),
            ground_enabled=bool(ctx.get("ground_enabled", True)),
            ground_height=float(ctx.get("ground_height") or 0.0),
            self_collisions=bool(ctx.get("self_collisions", False)),
            show_colliders=bool(ctx.get("show_colliders", False)),
            home_positions=dict(ctx.get("home_positions") or {}),
            rigid_bodies=list(ctx.get("rigid_bodies") or []),
            particle_fill=dict(ctx.get("particle_fill") or {}),
            convex_decomposition_patterns=list(ctx.get("convex_decomposition_patterns") or []),
            friction_overrides=dict(ctx.get("friction_overrides") or {}),
        )
        return {
            "ok": True,
            "scene": scene,
            "report": (
                f"Newton USD scene ready: {scene['asset_path']} at {scene['root_path']}; "
                f"{len(scene['rigid_bodies'])} added rigid body/bodies; "
                f"{int(dict(scene.get('particle_fill') or {}).get('particle_count') or 0)} particles"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "scene": {}, "report": f"Newton USD scene FAILED: {type(exc).__name__}: {exc}"}


@node(
    name="NewtonViewerConfig",
    component="runtime",
    category=_CATEGORY,
    description="Select a registered Newton viewer provider. The editor can embed its browser/WebSocket URL.",
    inputs={
        "trigger": AnyPort,
        "provider": Text(default="viser"),
        "port": Int(default=8080),
        "label": Text(default="Blacknode Newton Viewer"),
        "background_color": Text(default="#6383c5"),
        "show_grid": Bool(default=True),
        "hdri": Enum(
            ["none", "apartment", "city", "dawn", "forest", "lobby", "night", "park", "studio", "sunset", "warehouse"],
            default="apartment",
        ),
        "show_hdri_background": Bool(default=True),
        "hdri_intensity": Float(default=1.0),
        "render_fps": Int(default=30),
        "camera_position": List(default=[]),
        "camera_target": List(default=[]),
        "camera_up_axis": Enum(["auto", "X", "Y", "Z"], default="auto"),
        "camera_speed": Float(default=1.0),
    },
    outputs={"ok": Bool, "viewer": Dict, "viewer_url": Text, "report": Text},
    primary_inputs=["trigger"],
    primary_outputs=["viewer", "viewer_url", "report"],
)
def newton_viewer_config(ctx: dict) -> dict:
    import math

    provider = str(ctx.get("provider") or "viser").strip().lower()
    port = int(ctx.get("port") or 8080)
    if not 1024 <= port <= 65535:
        return {"ok": False, "viewer": {}, "viewer_url": "", "report": "viewer port must be between 1024 and 65535"}
    background_color = str(ctx.get("background_color") or "#6383c5").strip()
    if (
        len(background_color) != 7
        or not background_color.startswith("#")
        or any(character not in "0123456789abcdefABCDEF" for character in background_color[1:])
    ):
        return {
            "ok": False, "viewer": {}, "viewer_url": "",
            "report": "background_color must be a six-digit hex color such as #6383c5",
        }

    def _optional_vec3(name: str) -> list[float]:
        values = list(ctx.get(name) or [])
        if not values:
            return []
        if len(values) != 3:
            raise ValueError(f"{name} must be empty for automatic framing or contain exactly three numbers")
        vector = [float(value) for value in values]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{name} must contain finite numbers")
        return vector

    try:
        camera_position = _optional_vec3("camera_position")
        camera_target = _optional_vec3("camera_target")
        if bool(camera_position) != bool(camera_target):
            raise ValueError("camera_position and camera_target must both be empty or both contain three numbers")
        camera_speed = float(ctx.get("camera_speed") or 1.0)
        if not math.isfinite(camera_speed) or camera_speed < 0.0:
            raise ValueError("camera_speed must be finite and nonnegative")
        hdri_intensity = float(
            ctx.get("hdri_intensity") if ctx.get("hdri_intensity") is not None else 1.0
        )
        if not math.isfinite(hdri_intensity) or not 0.0 <= hdri_intensity <= 10.0:
            raise ValueError("hdri_intensity must be finite and between 0 and 10")
        render_fps = int(ctx.get("render_fps") or 30)
        if not 1 <= render_fps <= 60:
            raise ValueError("render_fps must be between 1 and 60")
    except (TypeError, ValueError) as exc:
        return {"ok": False, "viewer": {}, "viewer_url": "", "report": str(exc)}

    camera_up_axis = str(ctx.get("camera_up_axis") or "auto").lower()
    hdri = str(ctx.get("hdri") or "apartment").strip().lower()
    viewer = {
        "kind": "blacknode.newton-viewer", "schema_version": 1,
        "provider": provider, "host": "0.0.0.0", "port": port,
        "label": str(ctx.get("label") or "Blacknode Newton Viewer"),
        "background_color": background_color.lower(),
        "show_grid": bool(ctx.get("show_grid", True)),
        "render_fps": render_fps,
        "environment": {
            "hdri": hdri,
            "show_background": bool(ctx.get("show_hdri_background", True)),
            "intensity": hdri_intensity,
        },
        "camera": {
            "position_m": camera_position,
            "target_m": camera_target,
            "up_axis": camera_up_axis,
            "speed_m_s": camera_speed,
        },
        "share": False,
    }
    url = f"http://127.0.0.1:{port}"
    return {
        "ok": True, "viewer": viewer, "viewer_url": url,
        "report": (
            f"{provider} viewer requested at {url}; server starts with NewtonSimulation "
            "and selects a free loopback port if that address is occupied"
        ),
    }


@node(
    name="NewtonSimulation",
    component="runtime",
    live=True,
    category=_CATEGORY,
    description="Run a validated USD scene in Newton XPBD with a swappable viewer. Motion starts disarmed.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["status", "start", "arm", "disarm", "pause", "resume", "reset", "stop"], default="status"),
        "run_id": Text(default="newton-scene"),
        "scene": Dict(default={}),
        "viewer": Dict(default={}),
        "device": Text(default="auto"),
        "fps": Int(default=60),
        "substeps": Int(default=4),
        "solver_iterations": Int(default=16),
        "joint_stiffness": Float(default=runtime.XPBD_DRIVE_STIFFNESS),
        "joint_damping": Float(default=runtime.XPBD_DRIVE_DAMPING),
        "joint_drive_overrides": Dict(default={}),
        "max_velocity_deg_s": Float(default=45.0),
        "max_step_deg": Float(default=2.0),
    },
    outputs={
        "ok": Bool, "running": Bool, "armed": Bool, "phase": Text,
        "session": Dict, "positions": Dict, "rigid_bodies": Dict,
        "viewer_url": Text, "report": Text,
    },
    primary_inputs=["trigger", "scene", "viewer"],
    primary_outputs=["session", "viewer_url", "report"],
)
def newton_simulation(ctx: dict) -> dict:
    run_id = str(ctx.get("run_id") or "newton-scene").strip() or "newton-scene"
    action = str(ctx.get("action") or "status").lower()
    try:
        if action == "start":
            status = runtime.start_session(
                run_id,
                dict(ctx.get("scene") or {}),
                dict(ctx.get("viewer") or {}),
                str(ctx.get("device") or "auto"),
                int(ctx.get("fps") or 60),
                int(ctx.get("substeps") or 4),
                int(ctx.get("solver_iterations") or 16),
                float(
                    ctx.get("joint_stiffness")
                    if ctx.get("joint_stiffness") is not None
                    else runtime.XPBD_DRIVE_STIFFNESS
                ),
                float(
                    ctx.get("joint_damping")
                    if ctx.get("joint_damping") is not None
                    else runtime.XPBD_DRIVE_DAMPING
                ),
                dict(ctx.get("joint_drive_overrides") or {}),
                float(ctx.get("max_velocity_deg_s") or 45.0),
                float(ctx.get("max_step_deg") or 2.0),
            )
        else:
            status = runtime.control_session(run_id, action)
        ok = not bool(status.get("last_error")) and status.get("phase") != "fault"
        return {
            "ok": ok,
            "running": bool(status.get("running")),
            "armed": bool(status.get("armed")),
            "phase": str(status.get("phase") or "unknown"),
            "session": status,
            "positions": dict(status.get("positions") or {}),
            "rigid_bodies": dict(status.get("rigid_body_positions_m") or {}),
            "viewer_url": str(status.get("viewer_url") or ""),
            "report": (
                f"Newton {status.get('phase')}; {'ARMED' if status.get('armed') else 'simulation motion disarmed'}; "
                f"viewer {status.get('viewer_url') or 'not started'}"
                + (f"; {status['last_error']}" if status.get("last_error") else "")
            ),
        }
    except Exception as exc:  # noqa: BLE001
        status = {**runtime.session_status(run_id), "last_error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": False, "running": bool(status.get("running")), "armed": False,
            "phase": str(status.get("phase") or "fault"), "session": status,
            "positions": dict(status.get("positions") or {}),
            "rigid_bodies": dict(status.get("rigid_body_positions_m") or {}),
            "viewer_url": str(status.get("viewer_url") or ""),
            "report": f"Newton simulation FAILED: {type(exc).__name__}: {exc}",
        }


@node(
    name="NewtonJointCommand",
    component="runtime",
    live=True,
    category=_CATEGORY,
    description="Send named articulation targets through the armed, joint-limited Newton command boundary.",
    inputs={
        "trigger": AnyPort,
        "run_id": Text(default="newton-scene"),
        "positions": Dict(default={}),
        "units": Enum(["radians", "degrees"], default="radians"),
        "source": Text(default="blacknode"),
    },
    outputs={"ok": Bool, "status": Dict, "applied": Dict, "clamped": Dict, "report": Text},
    primary_inputs=["trigger", "positions"],
    primary_outputs=["applied", "report"],
)
def newton_joint_command(ctx: dict) -> dict:
    import math

    run_id = str(ctx.get("run_id") or "newton-scene").strip() or "newton-scene"
    positions = dict(ctx.get("positions") or {})
    if str(ctx.get("units") or "radians") == "degrees":
        session = runtime.get_session(run_id)
        joint_units = dict(getattr(session, "joint_units", {}) or {})
        positions = {
            name: math.radians(float(value)) if joint_units.get(name, "radians") == "radians" else float(value)
            for name, value in positions.items()
        }
    try:
        status = runtime.command_session(run_id, positions, str(ctx.get("source") or "blacknode"))
        clamped = {name: status["targets"][name] for name in status.get("clamped") or []}
        return {
            "ok": True, "status": status,
            "applied": dict(status.get("targets") or {}), "clamped": clamped,
            "report": f"Accepted joint target {status.get('command_count')} for {run_id}"
            + (f"; clamped {', '.join(clamped)}" if clamped else ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "status": runtime.session_status(run_id), "applied": {}, "clamped": {},
            "report": f"Newton command rejected: {type(exc).__name__}: {exc}",
        }
