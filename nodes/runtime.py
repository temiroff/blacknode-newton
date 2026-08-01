"""Managed Newton XPBD sessions used by the Blacknode live nodes."""
from __future__ import annotations

import atexit
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .viewer_contract import available_viewers, create_viewer

# XPBD drive compliance is expressed on the solver scale. These values were
# qualified at 60 Hz / four substeps to hold a small articulation against
# gravity while retaining contact response.
XPBD_DRIVE_STIFFNESS = 1.0e8
XPBD_DRIVE_DAMPING = 1.0e5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_RUN_ID = "__blacknode_newton_workspace__"


def package_asset_uri(relative_path: str) -> str:
    return f"package://blacknode-newton/{relative_path.lstrip('/')}"


def resolve_asset_path(value: str) -> Path:
    raw = str(value or "").strip()
    prefix = "package://blacknode-newton/"
    if raw.startswith(prefix):
        path = (PACKAGE_ROOT / raw[len(prefix) :]).resolve()
        try:
            path.relative_to(PACKAGE_ROOT)
        except ValueError as exc:
            raise ValueError("package asset path escapes blacknode-newton") from exc
    else:
        path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"USD asset does not exist: {path}")
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"Newton scene asset must be USD, got: {path.suffix}")
    return path


def make_usd_scene_spec(
    *,
    asset_path: str,
    root_path: str,
    fixed_base: bool,
    ground_enabled: bool,
    ground_height: float,
    self_collisions: bool,
    show_colliders: bool,
    home_positions: dict[str, Any] | None,
    rigid_bodies: list[Any] | None,
    convex_decomposition_patterns: list[Any] | None,
    friction_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    path = resolve_asset_path(asset_path)
    clean_root = str(root_path or "/").strip() or "/"
    if not clean_root.startswith("/") or ".." in clean_root.split("/"):
        raise ValueError("root_path must be an absolute USD prim path")
    clean_ground_height = float(ground_height)
    if not math.isfinite(clean_ground_height):
        raise ValueError("ground_height must be finite")
    clean_home: dict[str, float] = {}
    for raw_name, raw_value in dict(home_positions or {}).items():
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("home position names cannot be empty")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"home pose for {name} must be finite")
        clean_home[name] = value

    clean_bodies: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(list(rigid_bodies or [])):
        if not isinstance(raw, dict):
            raise ValueError(f"rigid_bodies[{index}] must be an object")
        name = str(raw.get("name") or f"rigid_body_{index + 1}").strip()
        if not name or name in names:
            raise ValueError(f"rigid body name must be non-empty and unique: {name!r}")
        names.add(name)
        shape = str(raw.get("shape") or "box").strip().lower()
        if shape != "box":
            raise ValueError(f"rigid body {name!r} uses unsupported shape {shape!r}; supported: box")
        position = [float(value) for value in list(raw.get("position_m") or [0.0, 0.0, 0.0])]
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError(f"rigid body {name!r} position_m must contain three finite numbers")
        size_raw = raw.get("size_m", 0.05)
        sizes = (
            [float(size_raw)] * 3
            if isinstance(size_raw, (int, float))
            else [float(value) for value in list(size_raw)]
        )
        if len(sizes) != 3 or not all(math.isfinite(value) and 0.005 <= value <= 10.0 for value in sizes):
            raise ValueError(f"rigid body {name!r} size_m must contain three values from 0.005 to 10 metres")
        mass = float(raw.get("mass_kg", 0.1))
        friction = float(raw.get("friction", 0.8))
        if not math.isfinite(mass) or not 0.001 <= mass <= 10_000.0:
            raise ValueError(f"rigid body {name!r} mass_kg must be between 0.001 and 10000")
        if not math.isfinite(friction) or not 0.0 <= friction <= 10.0:
            raise ValueError(f"rigid body {name!r} friction must be between 0 and 10")
        color = [float(value) for value in list(raw.get("color_rgb") or [0.7, 0.7, 0.7])]
        if len(color) != 3 or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in color):
            raise ValueError(f"rigid body {name!r} color_rgb must contain three values from 0 to 1")
        clean_bodies.append({
            "name": name,
            "shape": shape,
            "position_m": position,
            "size_m": sizes,
            "mass_kg": mass,
            "friction": friction,
            "color_rgb": color,
        })

    clean_patterns = []
    for raw in list(convex_decomposition_patterns or []):
        pattern = str(raw or "").strip()
        if pattern and pattern not in clean_patterns:
            clean_patterns.append(pattern)
    clean_friction: dict[str, float] = {}
    for raw_pattern, raw_mu in dict(friction_overrides or {}).items():
        pattern = str(raw_pattern or "").strip()
        mu = float(raw_mu)
        if not pattern or not math.isfinite(mu) or not 0.0 <= mu <= 10.0:
            raise ValueError("friction_overrides must map non-empty path patterns to values from 0 to 10")
        clean_friction[pattern] = mu
    return {
        "kind": "blacknode.newton-scene",
        "schema_version": 2,
        "asset_path": str(path),
        "source_asset": str(asset_path),
        "root_path": clean_root,
        "fixed_base": bool(fixed_base),
        "home_positions": clean_home,
        "rigid_bodies": clean_bodies,
        "ground": {"enabled": bool(ground_enabled), "height_m": clean_ground_height},
        "self_collisions": bool(self_collisions),
        "render": {"show_colliders": bool(show_colliders)},
        "collision": {
            "convex_decomposition_patterns": clean_patterns,
            "friction_overrides": clean_friction,
        },
    }


def make_empty_scene_spec() -> dict[str, Any]:
    """Create the real, asset-free stage used by the Newton editor workspace."""
    return {
        "kind": "blacknode.newton-scene",
        "schema_version": 2,
        "asset_path": "",
        "source_asset": "",
        "root_path": "/",
        "fixed_base": True,
        "home_positions": {},
        "rigid_bodies": [],
        "ground": {"enabled": True, "height_m": 0.0},
        "self_collisions": False,
        "render": {"show_colliders": False},
        "collision": {
            "convex_decomposition_patterns": [],
            "friction_overrides": {},
        },
    }


class NewtonSession:
    """One real-time USD scene, solver, viewer, and safe command boundary."""

    def __init__(
        self,
        run_id: str,
        scene: dict[str, Any],
        viewer_config: dict[str, Any],
        device: str,
        fps: int,
        substeps: int,
        solver_iterations: int,
        joint_stiffness: float,
        joint_damping: float,
        joint_drive_overrides: dict[str, Any],
        max_velocity_deg_s: float,
        max_step_deg: float,
    ) -> None:
        if scene.get("kind") != "blacknode.newton-scene":
            raise ValueError("connect a blacknode.newton-scene")
        if viewer_config.get("kind") != "blacknode.newton-viewer":
            raise ValueError("connect a blacknode.newton-viewer")
        self.run_id = run_id
        self.scene = dict(scene)
        self.viewer_config = dict(viewer_config)
        self.device_request = str(device or "auto")
        self.fps = max(10, min(120, int(fps)))
        self.substeps = max(1, min(16, int(substeps)))
        self.solver_iterations = max(1, min(64, int(solver_iterations)))
        self.joint_stiffness = self._validate_drive_gain("joint_stiffness", joint_stiffness)
        self.joint_damping = self._validate_drive_gain("joint_damping", joint_damping)
        self.joint_drive_overrides = self._validate_drive_overrides(joint_drive_overrides)
        self.joint_drive_gains: dict[str, dict[str, float]] = {}
        self.max_velocity_rad_s = math.radians(max(0.1, float(max_velocity_deg_s)))
        self.max_step_rad = math.radians(max(0.01, float(max_step_deg)))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.sim_time = 0.0
        self.frame_count = 0
        self.command_count = 0
        self.last_command_at = 0.0
        self.last_command_source = ""
        self.armed = False
        self.paused = False
        self.phase = "starting"
        self.last_error = ""
        self.viewer: Any = None
        self.model: Any = None
        self.solver: Any = None
        self.control: Any = None
        self.contacts: Any = None
        self.state_0: Any = None
        self.state_1: Any = None
        self.rigid_body_indices: dict[str, int] = {}
        self.authored_dynamic_body_names: list[str] = []
        self.active_collision_shapes = 0
        self.usd_mesh_count = 0
        self.usd_meshes_with_normals = 0
        self.usd_collision_mesh_count = 0
        self.joint_indices: dict[str, int] = {}
        self.joint_limits: dict[str, tuple[float, float]] = {}
        self.joint_units: dict[str, str] = {}
        self.current: dict[str, float] = {}
        self.desired: dict[str, float] = {}
        self.applied: dict[str, float] = {}
        self.home: dict[str, float] = {}
        self.clamped: list[str] = []
        self.reset_requested = False
        self.startup_config = {
            "device": self.device_request,
            "fps": self.fps,
            "substeps": self.substeps,
            "solver_iterations": self.solver_iterations,
            "joint_stiffness": self.joint_stiffness,
            "joint_damping": self.joint_damping,
            "joint_drive_overrides": self.joint_drive_overrides,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "max_step_rad": self.max_step_rad,
        }

    @staticmethod
    def _validate_drive_gain(name: str, value: Any) -> float:
        gain = float(value)
        if not math.isfinite(gain) or gain < 0.0 or gain > 1.0e12:
            raise ValueError(f"{name} must be finite and between 0 and 1e12")
        return gain

    @classmethod
    def _validate_drive_overrides(cls, overrides: dict[str, Any]) -> dict[str, dict[str, float]]:
        clean: dict[str, dict[str, float]] = {}
        for raw_name, raw_settings in dict(overrides or {}).items():
            name = str(raw_name or "").strip()
            if not name or not isinstance(raw_settings, dict):
                raise ValueError("joint_drive_overrides must map joint names to settings objects")
            unknown_fields = sorted(set(raw_settings) - {"stiffness", "damping"})
            if unknown_fields:
                raise ValueError(
                    f"joint drive override for {name!r} has unsupported fields: {', '.join(unknown_fields)}"
                )
            settings: dict[str, float] = {}
            if "stiffness" in raw_settings:
                settings["stiffness"] = cls._validate_drive_gain(
                    f"joint_drive_overrides[{name!r}].stiffness", raw_settings["stiffness"]
                )
            if "damping" in raw_settings:
                settings["damping"] = cls._validate_drive_gain(
                    f"joint_drive_overrides[{name!r}].damping", raw_settings["damping"]
                )
            clean[name] = settings
        return clean

    def _imports(self):
        try:
            import newton
            import warp as wp
        except Exception as exc:  # pragma: no cover - package health normally catches this
            raise RuntimeError("Newton and Warp are required; install package prerequisites") from exc
        return newton, wp

    def _resolve_device(self, wp: Any) -> str:
        requested = self.device_request.lower()
        if requested == "auto":
            return "cuda:0" if wp.is_cuda_available() else "cpu"
        if requested == "cuda":
            if not wp.is_cuda_available():
                raise RuntimeError("CUDA was requested but Warp cannot access a CUDA device")
            return "cuda:0"
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda:"):
            wp.get_device(requested)
            return requested
        raise ValueError("device must be auto, cpu, cuda, or a CUDA device such as cuda:0")

    def _build(self) -> None:
        newton, wp = self._imports()
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        except Exception as exc:  # pragma: no cover - declared importer dependency
            raise RuntimeError("OpenUSD Python schemas are required to load USD scenes") from exc
        device = self._resolve_device(wp)
        collision_config = dict(self.scene.get("collision") or {})
        decomposition_patterns = [
            str(value) for value in collision_config.get("convex_decomposition_patterns") or []
        ]
        root_path = str(self.scene.get("root_path") or "/").rstrip("/") or "/"
        normalized_collision_paths: set[str] = set()
        builder = newton.ModelBuilder()
        stage = None
        asset_path = str(self.scene.get("asset_path") or "").strip()

        def collision_authored(prim: Any) -> bool:
            cursor = prim
            while cursor and cursor.IsValid():
                if cursor.HasAPI(UsdPhysics.CollisionAPI) or "/collisions/" in str(cursor.GetPath()):
                    return True
                cursor = cursor.GetParent()
            return False

        collision_meshes = 0
        if asset_path:
            stage = Usd.Stage.Open(asset_path)
            if stage is None:
                raise RuntimeError(f"OpenUSD could not open {asset_path}")
            # Some USD authoring tools place collision APIs on an Xform above
            # the mesh. Newton consumes them on the mesh itself. Normalize only
            # the in-memory stage so the source file stays unchanged.
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if root_path != "/" and path != root_path and not path.startswith(root_path + "/"):
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    self.usd_mesh_count += 1
                    normals = UsdGeom.Mesh(prim).GetNormalsAttr().Get()
                    if normals is not None and len(normals) > 0:
                        self.usd_meshes_with_normals += 1
                    if collision_authored(prim):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        approximation = (
                            "convexDecomposition"
                            if any(pattern in path for pattern in decomposition_patterns)
                            else "convexHull"
                        )
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
                        collision_meshes += 1
                        normalized_collision_paths.add(path)
                if prim.HasRelationship("material:binding"):
                    UsdShade.MaterialBindingAPI.Apply(prim)
            self.usd_collision_mesh_count = collision_meshes
            # Preserve concave geometry with a coarse, deterministic
            # decomposition when the operator explicitly requests it.
            builder.default_mesh_approximation_cfg.coacd_threshold = 0.2
            builder.add_usd(
                stage,
                root_path=str(self.scene.get("root_path") or "/"),
                # None preserves authored fixed/free joints. Passing False
                # fixes every rigid body in the imported stage, including props.
                floating=None if bool(self.scene.get("fixed_base", True)) else True,
                enable_self_collisions=bool(self.scene.get("self_collisions", False)),
                load_visual_shapes=True,
                hide_collision_shapes=not bool(
                    dict(self.scene.get("render") or {}).get("show_colliders", False)
                ),
                force_show_colliders=bool(
                    dict(self.scene.get("render") or {}).get("show_colliders", False)
                ),
                force_position_velocity_actuation=True,
            )
        for joint_id, joint_type in enumerate(builder.joint_type):
            if int(joint_type) != int(newton.JointType.FREE):
                continue
            body_index = int(builder.joint_child[joint_id])
            label = str(builder.body_label[body_index] or f"body_{body_index}")
            short_name = label.rstrip("/").rsplit("/", 1)[-1] or f"body_{body_index}"
            name = short_name
            if name in self.rigid_body_indices:
                name = label.strip("/").replace("/", ":") or f"body_{body_index}"
            self.rigid_body_indices[name] = body_index
            self.authored_dynamic_body_names.append(name)
        friction_overrides = dict(collision_config.get("friction_overrides") or {})
        for shape_index, label in enumerate(builder.shape_label):
            text_label = str(label)
            for pattern, mu in friction_overrides.items():
                if str(pattern) in text_label:
                    builder.shape_material_mu[shape_index] = float(mu)
                    builder.shape_material_mu_torsional[shape_index] = 0.05
                    builder.shape_material_mu_rolling[shape_index] = 0.01
                    break
        labels = list(builder.joint_label)
        starts = list(builder.joint_q_start)
        home_positions = dict(self.scene.get("home_positions") or self.scene.get("home_radians") or {})
        for joint_id, (label, joint_type) in enumerate(zip(labels, builder.joint_type)):
            if int(joint_type) not in {int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)}:
                continue
            name = str(label).rsplit("/", 1)[-1]
            if not name or name in self.joint_indices:
                raise RuntimeError(
                    f"USD teleoperation requires unique one-DOF joint names; duplicate: {name!r}"
                )
            index = int(starts[joint_id])
            self.joint_indices[name] = index
            lower = float(builder.joint_limit_lower[index])
            upper = float(builder.joint_limit_upper[index])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise RuntimeError(
                    f"joint {name!r} needs finite ordered limits for safe teleoperation, got [{lower}, {upper}]"
                )
            self.joint_limits[name] = (lower, upper)
            self.joint_units[name] = (
                "radians" if int(joint_type) == int(newton.JointType.REVOLUTE) else "metres"
            )
            initial = min(upper, max(lower, float(home_positions.get(name, builder.joint_q[index]))))
            builder.joint_q[index] = initial
            builder.joint_target_q[index] = initial
            override = self.joint_drive_overrides.get(name, {})
            stiffness = float(override.get("stiffness", self.joint_stiffness))
            damping = float(override.get("damping", self.joint_damping))
            builder.joint_target_ke[index] = stiffness
            builder.joint_target_kd[index] = damping
            self.joint_drive_gains[name] = {"stiffness": stiffness, "damping": damping}
            self.current[name] = initial
            self.desired[name] = initial
            self.applied[name] = initial
            self.home[name] = initial
        unknown_home = sorted(set(home_positions) - set(self.joint_indices))
        if unknown_home:
            raise RuntimeError("home_positions contains unknown one-DOF joints: " + ", ".join(unknown_home))
        unknown_drives = sorted(set(self.joint_drive_overrides) - set(self.joint_indices))
        if unknown_drives:
            raise RuntimeError(
                "joint_drive_overrides contains unknown one-DOF joints: " + ", ".join(unknown_drives)
            )

        rigid_bodies = list(self.scene.get("rigid_bodies") or [])
        for body_spec in rigid_bodies:
            name = str(body_spec["name"])
            position = tuple(float(value) for value in body_spec["position_m"])
            sizes = tuple(float(value) for value in body_spec["size_m"])
            mass = float(body_spec["mass_kg"])
            body_index = builder.add_body(
                xform=wp.transform(position, wp.quat_identity()), label=name
            )
            self.rigid_body_indices[name] = body_index
            density = mass / (sizes[0] * sizes[1] * sizes[2])
            body_cfg = newton.ModelBuilder.ShapeConfig(
                density=density, mu=float(body_spec.get("friction", 0.8)), restitution=0.0
            )
            builder.add_shape_box(
                body_index,
                hx=sizes[0] / 2.0,
                hy=sizes[1] / 2.0,
                hz=sizes[2] / 2.0,
                cfg=body_cfg,
                color=tuple(float(value) for value in body_spec.get("color_rgb") or [0.7] * 3),
                label=f"{name}_shape",
            )
        ground = dict(self.scene.get("ground") or {})
        if ground.get("enabled", "ground_height_m" in self.scene):
            builder.add_ground_plane(
                height=float(ground.get("height_m", self.scene.get("ground_height_m") or 0.0))
            )
        self.model = builder.finalize(device=device)
        flags = self.model.shape_flags.numpy().tolist()
        collision_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.active_collision_shapes = sum(
            1 for label, flag in zip(self.model.shape_label, flags)
            if any(str(label).startswith(path) for path in normalized_collision_paths)
            and int(flag) & collision_bit
        )
        if collision_meshes and self.active_collision_shapes < collision_meshes:
            raise RuntimeError(
                "Newton did not activate every normalized USD collision mesh; physics cannot start safely"
            )
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=self.solver_iterations)
        self.contacts = self.model.contacts()
        self.control = self.model.control()
        self._new_states(newton, wp)
        provider = str(self.viewer_config.get("provider") or "viser")
        self.viewer = create_viewer(provider, self, self.model, self.viewer_config)

    def _new_states(self, newton: Any, wp: Any) -> None:
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        if self.control is None or self.control.joint_target_q is None:
            self.sim_time = 0.0
            self.frame_count = 0
            return
        targets = self.model.joint_target_q.numpy().tolist()
        for name, index in self.joint_indices.items():
            targets[index] = self.applied[name]
        self.control.joint_target_q.assign(
            wp.array(targets, dtype=wp.float32, device=self.model.device)
        )
        self.control.joint_target_qd.zero_()
        self.sim_time = 0.0
        self.frame_count = 0

    def start(self) -> dict[str, Any]:
        self._build()
        self.phase = "running"
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name=f"blacknode-newton-{self.run_id}"
        )
        self.thread.start()
        return self.status()

    def _apply_safe_target(self, frame_dt: float, wp: Any) -> None:
        if self.control is None or self.control.joint_target_q is None:
            return
        target_array = self.control.joint_target_q.numpy().tolist()
        max_delta = min(self.max_step_rad, self.max_velocity_rad_s * frame_dt)
        with self.lock:
            for name, index in self.joint_indices.items():
                requested = self.desired[name]
                prior = self.applied[name]
                delta = min(max_delta, max(-max_delta, requested - prior))
                value = prior + delta
                lower, upper = self.joint_limits[name]
                value = min(upper, max(lower, value))
                self.applied[name] = value
                target_array[index] = value
        self.control.joint_target_q.assign(
            wp.array(target_array, dtype=wp.float32, device=self.model.device)
        )

    def _read_state(self, newton: Any) -> None:
        if self.state_0.joint_q is None:
            return
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        coordinates = self.state_0.joint_q.numpy().tolist()
        with self.lock:
            self.current = {name: float(coordinates[index]) for name, index in self.joint_indices.items()}

    def _loop(self) -> None:
        newton, wp = self._imports()
        frame_dt = 1.0 / self.fps
        step_dt = frame_dt / self.substeps
        next_frame = time.perf_counter()
        try:
            while not self.stop_event.is_set() and self.viewer.is_running():
                with self.lock:
                    paused = self.paused
                    reset = self.reset_requested
                    self.reset_requested = False
                if reset:
                    self._new_states(newton, wp)
                if not paused:
                    self._apply_safe_target(frame_dt, wp)
                    for _ in range(self.substeps):
                        self.state_0.clear_forces()
                        self.model.collide(self.state_0, self.contacts)
                        self.solver.step(
                            self.state_0, self.state_1, self.control, self.contacts, step_dt
                        )
                        self.state_0, self.state_1 = self.state_1, self.state_0
                    self.sim_time += frame_dt
                    self.frame_count += 1
                    self._read_state(newton)
                self.viewer.begin_frame(self.sim_time)
                self.viewer.log_state(self.state_0)
                self.viewer.end_frame()
                next_frame += frame_dt
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_frame = time.perf_counter()
        except Exception as exc:  # noqa: BLE001 - captured for node status
            with self.lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.phase = "fault"
                self.armed = False
        finally:
            if self.phase != "fault":
                self.phase = "stopped"
            self.armed = False
            if self.viewer is not None:
                self.viewer.close()

    def set_armed(self, armed: bool) -> dict[str, Any]:
        with self.lock:
            if self.phase not in {"running", "paused"}:
                raise RuntimeError("simulation is not running")
            self.armed = bool(armed)
            if self.armed:
                self.desired = dict(self.current)
                self.applied = dict(self.current)
            else:
                self.desired = dict(self.applied)
        return self.status()

    def command(self, positions: dict[str, Any], source: str = "blacknode") -> dict[str, Any]:
        with self.lock:
            if not self.armed:
                raise RuntimeError("simulation motion is disarmed; arm explicitly before commanding joints")
            if not positions:
                raise ValueError("positions must contain at least one articulation joint")
            unknown = sorted(set(positions) - set(self.joint_indices))
            if unknown:
                raise ValueError("unknown articulation joint(s): " + ", ".join(unknown))
            clamped: list[str] = []
            for name, raw in positions.items():
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"joint {name} target must be finite")
                lower, upper = self.joint_limits[name]
                safe = min(upper, max(lower, value))
                if safe != value:
                    clamped.append(name)
                self.desired[name] = safe
            self.clamped = clamped
            self.command_count += 1
            self.last_command_at = time.time()
            self.last_command_source = str(source)
        return {**self.status(), "command_source": str(source), "clamped": clamped}

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self.lock:
            self.paused = bool(paused)
            self.phase = "paused" if self.paused else "running"
        return self.status()

    def request_reset(self) -> dict[str, Any]:
        with self.lock:
            self.armed = False
            self.desired = dict(self.home)
            self.applied = dict(self.home)
            self.reset_requested = True
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            rigid_body_positions: dict[str, list[float]] = {}
            if self.state_0 is not None and self.rigid_body_indices:
                try:
                    body_q = self.state_0.body_q.numpy()
                    rigid_body_positions = {
                        name: [float(value) for value in body_q[index][:3]]
                        for name, index in self.rigid_body_indices.items()
                    }
                except Exception:
                    rigid_body_positions = {}
            running = bool(self.thread and self.thread.is_alive()) and not self.stop_event.is_set()
            return {
                "kind": "blacknode.newton-session",
                "schema_version": 1,
                "run_id": self.run_id,
                "running": running,
                "phase": self.phase,
                "armed": self.armed,
                "paused": self.paused,
                "viewer_provider": str(self.viewer_config.get("provider") or "viser"),
                "available_viewers": available_viewers(),
                "viewer_url": str(getattr(self.viewer, "url", "") or ""),
                "viewer_port": int(getattr(self.viewer, "port", 0) or 0),
                "viewer_requested_port": int(
                    getattr(self.viewer, "requested_port", self.viewer_config.get("port") or 0) or 0
                ),
                "device": str(getattr(self.model, "device", self.device_request)),
                "sim_time": self.sim_time,
                "frame_count": self.frame_count,
                "command_count": self.command_count,
                "last_command_at": self.last_command_at,
                "last_command_source": self.last_command_source,
                "joint_names": list(self.joint_indices),
                "joint_units": dict(self.joint_units),
                "positions": dict(self.current),
                "targets": dict(self.desired),
                "applied": dict(self.applied),
                "joint_limits": {name: list(bounds) for name, bounds in self.joint_limits.items()},
                "joint_drive_gains": {
                    name: dict(settings) for name, settings in self.joint_drive_gains.items()
                },
                "rigid_body_positions_m": rigid_body_positions,
                "authored_dynamic_body_names": list(self.authored_dynamic_body_names),
                "authored_dynamic_body_count": len(self.authored_dynamic_body_names),
                "usd_mesh_count": self.usd_mesh_count,
                "usd_meshes_with_normals": self.usd_meshes_with_normals,
                "usd_collision_mesh_count": self.usd_collision_mesh_count,
                "active_collision_shapes": self.active_collision_shapes,
                "clamped": list(self.clamped),
                "last_error": self.last_error,
                "elapsed_seconds": max(0.0, time.time() - self.started_at),
            }

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=3.0)
        if self.viewer is not None:
            self.viewer.close()
        self.armed = False
        if self.phase != "fault":
            self.phase = "stopped"
        return self.status()


_LOCK = threading.RLock()
_SESSIONS: dict[str, NewtonSession] = {}
_SHUTDOWN_HOOKS: dict[str, Any] = {}


def register_shutdown_hook(name: str, hook: Any) -> None:
    """Let optional components join the package's managed-service shutdown."""
    key = str(name or "").strip()
    if not key or not callable(hook):
        raise ValueError("shutdown hook requires a name and callable")
    with _LOCK:
        _SHUTDOWN_HOOKS[key] = hook


def get_session(run_id: str) -> NewtonSession | None:
    with _LOCK:
        return _SESSIONS.get(run_id)


def start_session(
    run_id: str,
    scene: dict[str, Any],
    viewer: dict[str, Any],
    device: str,
    fps: int,
    substeps: int,
    solver_iterations: int,
    joint_stiffness: float,
    joint_damping: float,
    joint_drive_overrides: dict[str, Any],
    max_velocity_deg_s: float,
    max_step_deg: float,
) -> dict[str, Any]:
    key = str(run_id or "").strip() or f"newton-{uuid.uuid4().hex[:8]}"
    with _LOCK:
        prior = _SESSIONS.get(key)
        if prior is not None and prior.status()["running"]:
            requested_startup_config = {
                "device": str(device or "auto"),
                "fps": max(10, min(120, int(fps))),
                "substeps": max(1, min(16, int(substeps))),
                "solver_iterations": max(1, min(64, int(solver_iterations))),
                "joint_stiffness": prior._validate_drive_gain("joint_stiffness", joint_stiffness),
                "joint_damping": prior._validate_drive_gain("joint_damping", joint_damping),
                "joint_drive_overrides": prior._validate_drive_overrides(joint_drive_overrides),
                "max_velocity_rad_s": math.radians(max(0.1, float(max_velocity_deg_s))),
                "max_step_rad": math.radians(max(0.01, float(max_step_deg))),
            }
            if (
                prior.scene == scene
                and prior.viewer_config == viewer
                and prior.startup_config == requested_startup_config
            ):
                return prior.status()
            raise RuntimeError(f"Newton session '{key}' is already running with different configuration")
        session = NewtonSession(
            key, scene, viewer, device, fps, substeps, solver_iterations,
            joint_stiffness, joint_damping, joint_drive_overrides,
            max_velocity_deg_s, max_step_deg,
        )
        _SESSIONS[key] = session
    try:
        return session.start()
    except Exception:
        with _LOCK:
            if _SESSIONS.get(key) is session:
                _SESSIONS.pop(key, None)
        session.stop()
        raise


def session_status(run_id: str) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        return {
            "kind": "blacknode.newton-session", "schema_version": 1,
            "run_id": run_id, "running": False, "phase": "stopped", "armed": False,
            "paused": False, "viewer_url": "", "available_viewers": available_viewers(),
            "viewer_port": 0, "viewer_requested_port": 0,
            "positions": {}, "targets": {}, "joint_limits": {},
            "joint_names": [], "joint_units": {}, "rigid_body_positions_m": {},
            "frame_count": 0, "command_count": 0, "last_error": "",
        }
    return session.status()


def control_session(run_id: str, action: str) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        if action == "stop":
            return session_status(run_id)
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    if action == "arm":
        return session.set_armed(True)
    if action == "disarm":
        return session.set_armed(False)
    if action == "pause":
        return session.set_paused(True)
    if action == "resume":
        return session.set_paused(False)
    if action == "reset":
        return session.request_reset()
    if action == "stop":
        status = session.stop()
        with _LOCK:
            _SESSIONS.pop(run_id, None)
        return status
    if action == "status":
        return session.status()
    raise ValueError(f"unsupported Newton session action: {action}")


def command_session(run_id: str, positions: dict[str, Any], source: str = "blacknode") -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    return session.command(positions, source)


_WORKSPACE_LOCK = threading.RLock()
_WORKSPACE_SCENE_PATH = ""
_WORKSPACE_VIEWER_PROVIDER = "viser"


def _workspace_viewer_config(provider: str = "viser") -> dict[str, Any]:
    return {
        "kind": "blacknode.newton-viewer",
        "schema_version": 1,
        "provider": str(provider or "viser").strip().lower(),
        "host": "0.0.0.0",
        "port": 8080,
        "label": "Blacknode Newton",
        "background_color": "#111827",
        "show_grid": True,
        "environment": {
            "hdri": "none",
            "show_background": True,
            "intensity": 1.0,
        },
        "camera": {
            "position_m": [],
            "target_m": [],
            "up_axis": "auto",
            "speed_m_s": 1.0,
        },
        "share": False,
    }


def _workspace_status() -> dict[str, Any]:
    status = session_status(WORKSPACE_RUN_ID)
    service_open = bool(status.get("running"))
    asset_path = _WORKSPACE_SCENE_PATH if service_open else ""
    dynamic_count = int(status.get("authored_dynamic_body_count") or 0)
    collision_count = int(status.get("usd_collision_mesh_count") or 0)
    mesh_count = int(status.get("usd_mesh_count") or 0)
    warning = ""
    if service_open and asset_path and mesh_count and collision_count == 0:
        warning = "This USD is visual-only; it has no authored mesh collision geometry."
    return {
        **status,
        "kind": "blacknode.newton-workspace",
        "schema_version": 1,
        "open": service_open,
        "simulation_running": service_open and not bool(status.get("paused")),
        "asset_path": asset_path,
        "scene_label": Path(asset_path).name if asset_path else "Empty stage",
        "dynamic_body_count": dynamic_count,
        "warning": warning,
    }


def _start_workspace(
    scene: dict[str, Any], asset_path: str = "", provider: str = "viser"
) -> dict[str, Any]:
    global _WORKSPACE_SCENE_PATH, _WORKSPACE_VIEWER_PROVIDER
    prior = get_session(WORKSPACE_RUN_ID)
    if prior is not None:
        prior.stop()
    _WORKSPACE_SCENE_PATH = str(asset_path or "")
    _WORKSPACE_VIEWER_PROVIDER = str(provider or "viser").strip().lower()
    try:
        start_session(
            WORKSPACE_RUN_ID,
            scene,
            _workspace_viewer_config(_WORKSPACE_VIEWER_PROVIDER),
            "auto",
            60,
            4,
            16,
            XPBD_DRIVE_STIFFNESS,
            XPBD_DRIVE_DAMPING,
            {},
            45.0,
            2.0,
        )
        control_session(WORKSPACE_RUN_ID, "pause")
        return _workspace_status()
    except Exception:
        _WORKSPACE_SCENE_PATH = ""
        _WORKSPACE_VIEWER_PROVIDER = "viser"
        raise


def control_workspace(action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Operate the node-independent Newton workspace used by the editor app."""
    global _WORKSPACE_SCENE_PATH, _WORKSPACE_VIEWER_PROVIDER
    command = str(action or "status").strip().lower()
    values = dict(payload or {})
    with _WORKSPACE_LOCK:
        if command == "status":
            return _workspace_status()
        if command == "open":
            if _workspace_status()["open"]:
                return _workspace_status()
            return _start_workspace(
                make_empty_scene_spec(), provider=str(values.get("provider") or "viser")
            )
        if command == "new":
            return _start_workspace(
                make_empty_scene_spec(), provider=str(values.get("provider") or _WORKSPACE_VIEWER_PROVIDER)
            )
        if command == "open_usd":
            source = str(values.get("asset_path") or "").strip()
            scene = make_usd_scene_spec(
                asset_path=source,
                root_path=str(values.get("root_path") or "/"),
                fixed_base=bool(values.get("fixed_base", True)),
                ground_enabled=bool(values.get("ground_enabled", True)),
                ground_height=float(values.get("ground_height") or 0.0),
                self_collisions=bool(values.get("self_collisions", False)),
                show_colliders=bool(values.get("show_colliders", False)),
                home_positions={},
                rigid_bodies=[],
                convex_decomposition_patterns=[],
                friction_overrides={},
            )
            return _start_workspace(
                scene,
                str(scene["asset_path"]),
                provider=str(values.get("provider") or _WORKSPACE_VIEWER_PROVIDER),
            )
        if command == "set_viewer":
            provider = str(values.get("provider") or "").strip().lower()
            if not provider:
                raise ValueError("viewer provider is required")
            current = get_session(WORKSPACE_RUN_ID)
            if current is None:
                return _start_workspace(make_empty_scene_spec(), provider=provider)
            return _start_workspace(dict(current.scene), _WORKSPACE_SCENE_PATH, provider=provider)
        if command in {"play", "start"}:
            if not _workspace_status()["open"]:
                _start_workspace(make_empty_scene_spec())
            control_session(WORKSPACE_RUN_ID, "resume")
            return _workspace_status()
        if command in {"stop", "pause"}:
            if _workspace_status()["open"]:
                control_session(WORKSPACE_RUN_ID, "pause")
            return _workspace_status()
        if command == "reset":
            if _workspace_status()["open"]:
                control_session(WORKSPACE_RUN_ID, "reset")
            return _workspace_status()
        if command == "close":
            session = get_session(WORKSPACE_RUN_ID)
            if session is not None:
                session.stop()
            _WORKSPACE_SCENE_PATH = ""
            _WORKSPACE_VIEWER_PROVIDER = "viser"
            return _workspace_status()
    raise ValueError(f"unsupported Newton workspace action: {command}")


def runtime_status() -> dict[str, Any]:
    with _LOCK:
        runs = [session.status() for session in _SESSIONS.values()]
    return {
        "ok": not any(run.get("phase") == "fault" for run in runs),
        "active": any(run.get("running") for run in runs),
        "managed_runs": runs,
        "streams": [],
        "detached_count": 0,
        "report": f"{sum(bool(run.get('running')) for run in runs)} Newton session(s) running",
    }


def stop_runtime_services() -> dict[str, Any]:
    global _WORKSPACE_SCENE_PATH
    with _LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
        hooks = list(_SHUTDOWN_HOOKS.items())
    _WORKSPACE_SCENE_PATH = ""
    hook_errors: list[str] = []
    for name, hook in hooks:
        try:
            hook()
        except Exception as exc:  # optional service failures must not strand physics sessions
            hook_errors.append(f"{name}: {type(exc).__name__}: {exc}")
    stopped = 0
    for session in sessions:
        session.stop()
        stopped += 1
    return {
        "ok": not hook_errors,
        "stopped": {"managed_runs": stopped, "streams": 0, "detached": 0},
        "report": f"Stopped {stopped} Newton session(s)"
        + (f"; shutdown hook errors: {'; '.join(hook_errors)}" if hook_errors else ""),
    }


atexit.register(stop_runtime_services)
