"""Managed Newton XPBD sessions used by the Blacknode live nodes."""
from __future__ import annotations

import atexit
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .viewer_contract import available_viewers, create_viewer

# XPBD drive compliance is expressed on the solver scale. These values were
# qualified at 60 Hz / four substeps to hold a small articulation against
# gravity while retaining contact response.
XPBD_DRIVE_STIFFNESS = 1.0e8
XPBD_DRIVE_DAMPING = 1.0e5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_SCENE = "assets/scenes/so101_tabletop.usd"
DEFAULT_PICK_CONVEX_DECOMPOSITION_PATTERNS: tuple[str, ...] = ()
DEFAULT_PICK_FRICTION_OVERRIDES = {
    "/env/green_cube": 3.0,
    "/env/blue_cube": 3.0,
    "/env/red_cube": 3.0,
}
DEFAULT_PICK_MASS_OVERRIDES = {
    "/env/green_cube": 0.03,
    "/env/blue_cube": 0.03,
    "/env/red_cube": 0.03,
    "/env/green_container": 0.03,
    "/env/blue_container": 0.03,
    "/env/red_container": 0.03,
}
DEFAULT_STATIC_COLLISION_BODY_PATTERNS: tuple[str, ...] = ()
DEFAULT_MESH_APPROXIMATION_OVERRIDES: dict[str, str] = {}
_DEFAULT_CONTAINER_PROXY_BOXES = (
    {"position_m": [0.0, 0.0, -0.01237], "size_m": [0.046, 0.046, 0.005]},
    {"position_m": [-0.02445, 0.0, 0.0], "size_m": [0.00315, 0.046, 0.02975]},
    {"position_m": [0.02445, 0.0, 0.0], "size_m": [0.00315, 0.046, 0.02975]},
    {"position_m": [0.0, -0.02445, 0.0], "size_m": [0.046, 0.00315, 0.02975]},
    {"position_m": [0.0, 0.02445, 0.0], "size_m": [0.046, 0.00315, 0.02975]},
)
DEFAULT_CONTAINER_COLLISION_PROXIES = tuple(
    {
        "body_path": f"/so101_new_calib/env/{color}_container",
        "source_shape_path": (
            f"/so101_new_calib/env/{color}_container/{color}_container"
        ),
        "friction": 1.5,
        "boxes": copy.deepcopy(list(_DEFAULT_CONTAINER_PROXY_BOXES)),
    }
    for color in ("green", "blue", "red")
)
# XPBD does not consume actuator effort limits. A compliant gripper position
# drive lets rigid contacts oppose the commanded close target instead of
# forcing the jaws through an object.
DEFAULT_GRIPPER_DRIVE = {"stiffness": 1.0e5, "damping": 1.0e3}
DEFAULT_GRIP_PAD_SPECS = (
    # The authored jaw meshes are concave and their convex approximations
    # bridge empty space. Replace only those inaccurate physics shapes with
    # body-local boxes aligned to the visible opposing fingertip faces.
    {
        "body_path": "/so101_new_calib/gripper_link",
        "source_shape_pattern": (
            "/gripper_link/collisions/wrist_roll_follower_so101_v1"
        ),
        "label": "/__BlacknodeGripPads/fixed",
        "position_m": [-0.01307484, -0.00045399, -0.08632979],
        "rotation_xyzw": [-0.01652895, 0.70359731, 0.01668677, 0.71021068],
        "size_m": [0.043, 0.012, 0.003],
        "margin_m": 0.0001,
        "friction": 1.2,
    },
    {
        "body_path": "/so101_new_calib/moving_jaw_so101_v1_link",
        "source_shape_pattern": (
            "/moving_jaw_so101_v1_link/collisions/moving_jaw_so101_v1"
        ),
        "label": "/__BlacknodeGripPads/moving",
        "position_m": [-0.02814949, -0.06289308, 0.01901818],
        # Working-pose orientation of the compliant face. The moving jaw pivots
        # about 20 degrees by a 25 mm grasp; this keeps the effective flat pad
        # seated on the prop instead of presenting a separating edge.
        "rotation_xyzw": [-0.42187043, 0.58678212, -0.56744371, 0.39461340],
        "size_m": [0.043, 0.012, 0.003],
        "margin_m": 0.0001,
        "friction": 1.2,
    },
)
WORKSPACE_RUN_ID = "__blacknode_newton_workspace__"
WORKSPACE_HDRI_PRESETS = {
    "none", "custom", "apartment", "city", "dawn", "forest", "lobby",
    "night", "park", "studio", "sunset", "warehouse",
}
WORKSPACE_GROUND_PATH = "/Blacknode/Ground"
WORKSPACE_GROUND_MATERIAL_PATH = "/BlacknodeOVRT/GroundMaterial"
WORKSPACE_LIGHTS_PATH = "/Blacknode/Lights"
WORKSPACE_KEY_LIGHT_PATH = f"{WORKSPACE_LIGHTS_PATH}/Key"
WORKSPACE_HDRI_LIGHT_PATH = f"{WORKSPACE_LIGHTS_PATH}/HDRI"
WORKSPACE_MATERIAL_ROOT = "/__BlacknodeMaterials"
COLLIDER_DISPLAY_ROOT = "/__BlacknodeColliderDisplay"
VISUAL_DISPLAY_ROOT = "/__BlacknodeVisualDisplay"
DIGITAL_TWIN_HISTORY_LIMIT = 240
DIGITAL_TWIN_HISTORY_INTERVAL_SECONDS = 0.05
MAX_PHYSICS_SUBSTEPS = 64
_XACRO_EXPANSION_LOCK = threading.RLock()
_XACRO_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_XACRO_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_XACRO_MISSING_ENVIRONMENT = re.compile(
    r"environment variable ['\"](?P<name>[^'\"]+)['\"] is not set",
    re.IGNORECASE,
)
_DIGITAL_TWIN_ARTIFACT_LOCK = threading.RLock()
_DIGITAL_TWIN_ARTIFACT_ID = re.compile(r"^newton-run-[a-f0-9]{20}$")


def _digital_twin_artifact_directory() -> Path:
    configured = str(os.environ.get("BLACKNODE_CONFIG_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve() / "newton-runs"
    repository_root = PACKAGE_ROOT.parents[1]
    if (repository_root / "editor-server").is_dir():
        return repository_root / ".blacknode" / "newton-runs"
    local_root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".blacknode")
    return local_root.expanduser().resolve() / "Blacknode" / "newton-runs"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_index() -> dict[str, dict[str, Any]]:
    path = _digital_twin_artifact_directory() / "index.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Newton run-artifact index could not be read: {exc}") from exc
    records = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("Newton run-artifact index has an invalid format")
    return {
        str(item["artifact_id"]): dict(item)
        for item in records
        if isinstance(item, dict) and _DIGITAL_TWIN_ARTIFACT_ID.fullmatch(
            str(item.get("artifact_id") or "")
        )
    }


def list_digital_twin_artifacts(limit: int = 20) -> list[dict[str, Any]]:
    with _DIGITAL_TWIN_ARTIFACT_LOCK:
        records = list(_artifact_index().values())
    records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return records[: max(1, min(100, int(limit)))]


def _read_digital_twin_artifact(artifact_id: Any) -> dict[str, Any]:
    clean_id = str(artifact_id or "").strip()
    if not _DIGITAL_TWIN_ARTIFACT_ID.fullmatch(clean_id):
        raise ValueError("invalid Newton run-artifact id")
    path = _digital_twin_artifact_directory() / f"{clean_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Newton run artifact does not exist: {clean_id}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Newton run artifact could not be read: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("kind") != "blacknode.newton-run-artifact":
        raise ValueError("selected file is not a Blacknode Newton run artifact")
    if int(artifact.get("schema_version") or 0) != 1:
        raise ValueError("unsupported Newton run-artifact schema version")
    return artifact


def _workspace_scene_path(path: Any) -> str:
    """Canonicalize renderer-private wrapper paths used by workspace commands."""
    value = str(path or "").strip()
    if value == "/BlacknodeOVRT/Ground" or value.startswith("/BlacknodeOVRT/Ground/"):
        return WORKSPACE_GROUND_PATH
    return value


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


def resolve_robot_description_path(value: str) -> Path:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"robot description does not exist: {path}")
    if path.suffix.lower() not in {".urdf", ".xacro"}:
        raise ValueError(f"robot description must be .urdf or .xacro, got: {path.suffix}")
    return path


def resolve_mjcf_path(value: str) -> Path:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
    if path.suffix.lower() not in {".xml", ".mjcf"}:
        raise ValueError(f"MuJoCo model must be .xml or .mjcf, got: {path.suffix}")
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"MuJoCo XML could not be read: {path.name}: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "mujoco":
        raise ValueError(f"XML scene is not a MuJoCo model: {path.name}")
    return path


def _mjcf_has_world_plane(path: Path, visited: set[Path] | None = None) -> bool:
    """Inspect an MJCF include tree for an authored world plane."""
    checked = set() if visited is None else visited
    resolved = path.resolve()
    if resolved in checked:
        return False
    checked.add(resolved)
    try:
        root = ET.parse(resolved).getroot()
    except (ET.ParseError, OSError):
        return False
    for worldbody in root.findall("worldbody"):
        if any(
            str(geom.get("type") or "").strip().lower() == "plane"
            for geom in worldbody.iter("geom")
        ):
            return True
    for include in root.iter("include"):
        raw_file = str(include.get("file") or "").strip()
        if not raw_file:
            continue
        included = (resolved.parent / raw_file).resolve()
        if included.is_file() and _mjcf_has_world_plane(included, checked):
            return True
    return False


def _mjcf_named_keyframe_qpos(
    path: Path,
    name: str = "home",
    visited: set[Path] | None = None,
) -> list[float]:
    """Read one named MJCF keyframe from a model and its include tree."""
    checked = set() if visited is None else visited
    resolved = path.resolve()
    if resolved in checked:
        return []
    checked.add(resolved)
    try:
        root = ET.parse(resolved).getroot()
    except (ET.ParseError, OSError):
        return []
    for keyframe in root.findall("keyframe"):
        for key in keyframe.findall("key"):
            if str(key.get("name") or "").strip() != name:
                continue
            try:
                values = [float(value) for value in str(key.get("qpos") or "").split()]
            except ValueError as exc:
                raise ValueError(
                    f"MuJoCo keyframe {name!r} has invalid qpos values in {resolved.name}"
                ) from exc
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"MuJoCo keyframe {name!r} needs finite qpos values in {resolved.name}"
                )
            return values
    for include in root.iter("include"):
        raw_file = str(include.get("file") or "").strip()
        if not raw_file:
            continue
        included = (resolved.parent / raw_file).resolve()
        if not included.is_file():
            continue
        values = _mjcf_named_keyframe_qpos(included, name, checked)
        if values:
            return values
    return []


def _local_ros_package_path(package_name: str, source_path: Path) -> Path | None:
    """Resolve a ROS package next to a Xacro source when ROS is not installed."""
    candidates: list[Path] = []
    for parent in (source_path.parent, *source_path.parents):
        if (parent / "package.xml").is_file():
            candidates.append(parent)
        if parent.name.lower() == "src":
            candidates.extend(
                manifest.parent for manifest in parent.rglob("package.xml")
            )
            break
    for variable in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        for raw in os.environ.get(variable, "").split(os.pathsep):
            if raw:
                candidates.append(Path(raw) / "share" / package_name)
    for raw in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if raw:
            root = Path(raw)
            candidates.extend((root, root / package_name))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        manifest = resolved / "package.xml"
        if not manifest.is_file():
            continue
        try:
            declared_name = str(ET.parse(manifest).getroot().findtext("name") or "").strip()
        except (ET.ParseError, OSError):
            continue
        if declared_name == package_name:
            return resolved
    return None


def _clean_xacro_environment(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("xacro_environment must be an object of environment variable names and values")
    clean: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name or "").strip()
        if not _XACRO_ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"invalid Xacro environment variable name: {name or '<empty>'}")
        text = str(raw_value)
        if "\x00" in text:
            raise ValueError(f"Xacro environment variable {name!r} cannot contain a null byte")
        clean[name] = text
    return clean


def _clean_xacro_arguments(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("xacro_arguments must be an object of argument names and values")
    clean: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name or "").strip()
        if not _XACRO_ARGUMENT_NAME.fullmatch(name):
            raise ValueError(f"invalid Xacro argument name: {name or '<empty>'}")
        text = str(raw_value)
        if "\x00" in text:
            raise ValueError(f"Xacro argument {name!r} cannot contain a null byte")
        clean[name] = text
    return clean


def _expanded_xacro_xml(
    path: Path,
    xacro_environment: dict[str, Any] | None = None,
    xacro_arguments: dict[str, Any] | None = None,
) -> str:
    try:
        import xacro
    except Exception as exc:  # pragma: no cover - declared package dependency
        raise RuntimeError(
            "Xacro support requires xacro>=2.1; update the blacknode-newton runtime component"
        ) from exc
    environment = _clean_xacro_environment(xacro_environment)
    arguments = _clean_xacro_arguments(xacro_arguments)

    def load_yaml_utf8(filename: str):
        try:
            import yaml
            for unit in xacro.ConstructUnits:
                yaml.SafeLoader.add_constructor(unit.value.tag, unit.constructor)
        except Exception as exc:
            raise RuntimeError("Xacro YAML support requires PyYAML") from exc
        resolved = xacro.abs_filename_spec(filename)
        xacro.filestack.append(resolved)
        try:
            with open(resolved, encoding="utf-8-sig") as stream:
                return xacro.YamlListWrapper.wrap(yaml.safe_load(stream))
        finally:
            xacro.filestack.pop()
            xacro.all_includes.append(resolved)

    # xacro delegates $(find package) to ament_index_python. Robot-description
    # source trees are also useful outside a sourced ROS installation, so fall
    # back to package.xml discovery around the selected file and workspace.
    with _XACRO_EXPANSION_LOCK:
        previous_environment = {name: os.environ.get(name) for name in environment}
        original_load_yaml = xacro.load_yaml
        xacro_symbols = xacro._global_symbols.get("xacro")
        original_symbol_load_yaml = (
            xacro_symbols.get("load_yaml") if isinstance(xacro_symbols, dict) else None
        )
        original_legacy_load_yaml = xacro._global_symbols.get("load_yaml")
        try:
            os.environ.update(environment)
            # YAML 1.2 streams are Unicode. Explicit UTF-8 keeps Xacro models
            # portable on Windows hosts whose process code page is not UTF-8.
            xacro.load_yaml = load_yaml_utf8
            if isinstance(xacro_symbols, dict):
                xacro_symbols["load_yaml"] = load_yaml_utf8
            xacro._global_symbols["load_yaml"] = load_yaml_utf8
            import xacro.substitution_args as substitution_args
            original_find = substitution_args._eval_find
            original_eval_find = substitution_args._eval_dict.get("find")

            def find_package(package_name: str) -> str:
                try:
                    found = str(original_find(package_name))
                    return Path(found).as_posix() if os.name == "nt" else found
                except Exception as original_error:
                    local = _local_ros_package_path(package_name, path)
                    if local is not None:
                        # Xacro substitutes this value inside Python expressions
                        # such as xacro.load_yaml('$(find package)/file.yaml').
                        # Windows backslashes would become escape sequences there.
                        return local.as_posix()
                    raise RuntimeError(
                        f"ROS package {package_name!r} was not found beside {path.name} "
                        "or in AMENT_PREFIX_PATH/ROS_PACKAGE_PATH"
                    ) from original_error

            substitution_args._eval_find = find_package
            substitution_args._eval_dict["find"] = find_package
            document = xacro.process_file(str(path), mappings=arguments)
        except Exception as exc:
            message = str(exc)
            missing = _XACRO_MISSING_ENVIRONMENT.search(message)
            if missing is not None:
                name = missing.group("name")
                raise ValueError(
                    f"Xacro requires environment variable '{name}'. "
                    "Enter a value in the Xacro environment dialog and retry."
                ) from exc
            raise ValueError(f"Xacro expansion failed for {path.name}: {message}") from exc
        finally:
            xacro.load_yaml = original_load_yaml
            if isinstance(xacro_symbols, dict):
                xacro_symbols["load_yaml"] = original_symbol_load_yaml
            xacro._global_symbols["load_yaml"] = original_legacy_load_yaml
            if "substitution_args" in locals() and "original_find" in locals():
                substitution_args._eval_find = original_find
                if original_eval_find is None:
                    substitution_args._eval_dict.pop("find", None)
                else:
                    substitution_args._eval_dict["find"] = original_eval_find
            for name, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
    # Component Xacros are often useful on their own even when their mounting
    # link is defined by a higher-level robot file. Add geometry-free anchors
    # for only those referenced links that the expanded fragment omits.
    robots = document.getElementsByTagName("robot")
    if robots:
        declared_links = {
            element.getAttribute("name").strip()
            for element in document.getElementsByTagName("link")
            if element.getAttribute("name").strip()
        }
        referenced_links = {
            element.getAttribute("link").strip()
            for tag_name in ("parent", "child")
            for element in document.getElementsByTagName(tag_name)
            if element.getAttribute("link").strip()
        }
        for name in sorted(referenced_links - declared_links):
            link = document.createElement("link")
            link.setAttribute("name", name)
            robots[0].appendChild(link)
            declared_links.add(name)
        child_links = {
            element.getAttribute("link").strip()
            for element in document.getElementsByTagName("child")
            if element.getAttribute("link").strip()
        }
        roots = sorted(declared_links - child_links)
        if len(roots) > 1:
            raise ValueError(
                "Xacro produced a disconnected robot with multiple root links: "
                f"{', '.join(roots)}. Check parent/child link names and Xacro arguments."
            )
    # Newton can parse expanded XML directly. Make file-relative mesh and
    # texture references absolute first because the expanded document no
    # longer carries its source filename into the URDF importer.
    for tag_name in ("mesh", "texture"):
        for element in document.getElementsByTagName(tag_name):
            if not element.hasAttribute("filename"):
                continue
            raw = element.getAttribute("filename").strip()
            if not raw or raw.startswith("$("):
                continue
            if raw.lower().startswith("file://"):
                raw = raw[7:]
                if raw.startswith("/") and len(raw) >= 3 and raw[2] == ":":
                    raw = raw[1:]
            elif raw.lower().startswith("package://"):
                package_reference = raw[len("package://") :]
                package_name, separator, relative = package_reference.partition("/")
                package_path = _local_ros_package_path(package_name, path)
                if not separator or package_path is None:
                    raise ValueError(
                        f"ROS package URI could not be resolved while expanding {path.name}: {raw}"
                    )
                raw = str(package_path / relative)
            elif "://" in raw:
                continue
            candidate = Path(raw.replace("\\", "/"))
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            element.setAttribute("filename", candidate.resolve().as_posix())
    xml = document.toxml()
    if "<robot" not in xml:
        raise ValueError(f"Xacro expansion did not produce a URDF robot: {path.name}")
    return xml


def make_robot_description_scene_spec(
    *,
    asset_path: str,
    fixed_base: bool,
    ground_enabled: bool,
    ground_height: float,
    self_collisions: bool,
    show_colliders: bool,
    xacro_environment: dict[str, Any] | None = None,
    xacro_arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve_robot_description_path(asset_path)
    clean_ground_height = float(ground_height)
    if not math.isfinite(clean_ground_height):
        raise ValueError("ground_height must be finite")
    return {
        "kind": "blacknode.newton-scene",
        "schema_version": 2,
        "asset_path": str(path),
        "source_asset": str(path),
        "asset_format": "urdf",
        "robot_description_format": path.suffix.lower().lstrip("."),
        "robot_description_xml": (
            _expanded_xacro_xml(path, xacro_environment, xacro_arguments)
            if path.suffix.lower() == ".xacro"
            else ""
        ),
        "root_path": "/",
        "fixed_base": bool(fixed_base),
        "home_positions": {},
        "rigid_bodies": [],
        "particle_fill": {},
        "ground": {"enabled": bool(ground_enabled), "height_m": clean_ground_height},
        "self_collisions": bool(self_collisions),
        "render": {"show_colliders": bool(show_colliders)},
        "collision": {
            "convex_decomposition_patterns": [],
            "friction_overrides": {},
        },
    }


def make_mjcf_scene_spec(
    *,
    asset_path: str,
    fixed_base: bool | None,
    ground_enabled: bool | None,
    ground_height: float,
    self_collisions: bool,
    show_colliders: bool,
) -> dict[str, Any]:
    path = resolve_mjcf_path(asset_path)
    clean_ground_height = float(ground_height)
    if not math.isfinite(clean_ground_height):
        raise ValueError("ground_height must be finite")
    generated_ground = (
        not _mjcf_has_world_plane(path)
        if ground_enabled is None
        else bool(ground_enabled)
    )
    return {
        "kind": "blacknode.newton-scene",
        "schema_version": 2,
        "asset_path": str(path),
        "source_asset": str(path),
        "asset_format": "mjcf",
        "robot_description_format": "mjcf",
        "mjcf_home_qpos": _mjcf_named_keyframe_qpos(path),
        # None preserves an authored <freejoint>; explicit booleans let API
        # clients override the model's base behavior.
        "fixed_base": None if fixed_base is None else bool(fixed_base),
        "home_positions": {},
        "rigid_bodies": [],
        "particle_fill": {},
        "ground": {"enabled": generated_ground, "height_m": clean_ground_height},
        "self_collisions": bool(self_collisions),
        "render": {"show_colliders": bool(show_colliders)},
        "collision": {
            "convex_decomposition_patterns": [],
            "friction_overrides": {},
        },
    }


def _apply_mjcf_keyframe_qpos(builder: Any, newton: Any, values: list[Any]) -> None:
    """Apply MuJoCo qpos values to Newton's coordinate layout."""
    qpos = [float(value) for value in values]
    if len(qpos) != len(builder.joint_q):
        raise RuntimeError(
            "MuJoCo home keyframe qpos count does not match the imported model: "
            f"{len(qpos)} values for {len(builder.joint_q)} coordinates"
        )
    # MJCF serializes free/ball quaternions as wxyz. Newton stores them xyzw.
    for joint_type, raw_start in zip(builder.joint_type, builder.joint_q_start):
        start = int(raw_start)
        if int(joint_type) == int(newton.JointType.FREE):
            quaternion_start = start + 3
        elif int(joint_type) == int(newton.JointType.BALL):
            quaternion_start = start
        else:
            continue
        w, x, y, z = qpos[quaternion_start : quaternion_start + 4]
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1.0e-12:
            raise RuntimeError("MuJoCo home keyframe contains a zero-length quaternion")
        qpos[quaternion_start : quaternion_start + 4] = [
            x / norm, y / norm, z / norm, w / norm,
        ]
    for index, value in enumerate(qpos):
        builder.joint_q[index] = value


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
    particle_fill: dict[str, Any] | None = None,
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

    default_pick_scene = path == resolve_asset_path(
        package_asset_uri(DEFAULT_WORKSPACE_SCENE)
    )
    clean_patterns = []
    requested_patterns = [
        *(DEFAULT_PICK_CONVEX_DECOMPOSITION_PATTERNS if default_pick_scene else ()),
        *list(convex_decomposition_patterns or []),
    ]
    for raw in requested_patterns:
        pattern = str(raw or "").strip()
        if pattern and pattern not in clean_patterns:
            clean_patterns.append(pattern)
    requested_friction = (
        dict(DEFAULT_PICK_FRICTION_OVERRIDES)
        if path == resolve_asset_path(package_asset_uri(DEFAULT_WORKSPACE_SCENE))
        else {}
    )
    requested_friction.update(dict(friction_overrides or {}))
    clean_friction: dict[str, float] = {}
    for raw_pattern, raw_mu in requested_friction.items():
        pattern = str(raw_pattern or "").strip()
        mu = float(raw_mu)
        if not pattern or not math.isfinite(mu) or not 0.0 <= mu <= 10.0:
            raise ValueError("friction_overrides must map non-empty path patterns to values from 0 to 10")
        clean_friction[pattern] = mu
    clean_particle_fill: dict[str, Any] = {}
    if particle_fill:
        if not isinstance(particle_fill, dict):
            raise ValueError("particle_fill must be an object")
        name = str(particle_fill.get("name") or "grain").strip()
        if not name:
            raise ValueError("particle_fill.name cannot be empty")
        dimensions = [int(value) for value in list(particle_fill.get("dimensions") or [])]
        if len(dimensions) != 3 or any(value < 1 or value > 256 for value in dimensions):
            raise ValueError("particle_fill.dimensions must contain three integers from 1 to 256")
        particle_count = math.prod(dimensions)
        if particle_count > 100_000:
            raise ValueError("particle_fill cannot exceed 100000 particles")
        position = [float(value) for value in list(particle_fill.get("position_m") or [])]
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("particle_fill.position_m must contain three finite numbers")
        radius = float(particle_fill.get("radius_m", 0.001))
        if not math.isfinite(radius) or not 0.00025 <= radius <= 0.05:
            raise ValueError("particle_fill.radius_m must be between 0.00025 and 0.05 metres")
        spacing_raw = particle_fill.get("spacing_m", radius * 2.08)
        spacing = (
            [float(spacing_raw)] * 3
            if isinstance(spacing_raw, (int, float))
            else [float(value) for value in list(spacing_raw)]
        )
        if len(spacing) != 3 or not all(math.isfinite(value) and value <= 0.5 for value in spacing):
            raise ValueError("particle_fill.spacing_m must contain three finite values no greater than 0.5 metres")
        jitter = float(particle_fill.get("jitter_m", 0.0))
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("particle_fill.jitter_m must be a non-negative finite number")
        if any(value - jitter < radius * 2.0 for value in spacing):
            raise ValueError("particle_fill spacing minus jitter must be at least two particle radii")
        mass = float(particle_fill.get("mass_kg", 3.2e-6))
        if not math.isfinite(mass) or not 1.0e-9 <= mass <= 1.0:
            raise ValueError("particle_fill.mass_kg must be between 1e-9 and 1 kilogram per particle")
        friction = float(particle_fill.get("friction", 0.55))
        cohesion = float(particle_fill.get("cohesion", 0.0))
        adhesion = float(particle_fill.get("adhesion", 0.0))
        max_velocity = float(particle_fill.get("max_velocity_m_s", 3.0))
        color = [
            float(value)
            for value in list(particle_fill.get("color_rgb") or [0.82, 0.58, 0.18])
        ]
        if not math.isfinite(friction) or not 0.0 <= friction <= 10.0:
            raise ValueError("particle_fill.friction must be between 0 and 10")
        if not math.isfinite(cohesion) or not 0.0 <= cohesion <= 1.0:
            raise ValueError("particle_fill.cohesion must be between 0 and 1 metre")
        if not math.isfinite(adhesion) or not 0.0 <= adhesion <= 1.0:
            raise ValueError("particle_fill.adhesion must be between 0 and 1 metre")
        if not math.isfinite(max_velocity) or not 0.1 <= max_velocity <= 100.0:
            raise ValueError("particle_fill.max_velocity_m_s must be between 0.1 and 100")
        if len(color) != 3 or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in color):
            raise ValueError("particle_fill.color_rgb must contain three finite values from 0 to 1")
        proxy: dict[str, Any] = {}
        raw_proxy = particle_fill.get("container_collision_proxy")
        if raw_proxy:
            if not isinstance(raw_proxy, dict):
                raise ValueError("particle_fill.container_collision_proxy must be an object")
            body_path = str(raw_proxy.get("body_path") or "").strip()
            source_shape_path = str(raw_proxy.get("source_shape_path") or "").strip()
            if not body_path.startswith("/") or ".." in body_path.split("/"):
                raise ValueError(
                    "particle_fill.container_collision_proxy.body_path must be an absolute USD prim path"
                )
            if not source_shape_path.startswith("/") or ".." in source_shape_path.split("/"):
                raise ValueError(
                    "particle_fill.container_collision_proxy.source_shape_path must be an absolute USD prim path"
                )
            raw_boxes = list(raw_proxy.get("boxes") or [])
            if not 1 <= len(raw_boxes) <= 32:
                raise ValueError(
                    "particle_fill.container_collision_proxy.boxes must contain 1 to 32 boxes"
                )
            boxes: list[dict[str, list[float]]] = []
            for box_index, raw_box in enumerate(raw_boxes):
                if not isinstance(raw_box, dict):
                    raise ValueError(
                        f"particle_fill.container_collision_proxy.boxes[{box_index}] must be an object"
                    )
                box_position = [float(value) for value in list(raw_box.get("position_m") or [])]
                box_size = [float(value) for value in list(raw_box.get("size_m") or [])]
                if len(box_position) != 3 or not all(math.isfinite(value) for value in box_position):
                    raise ValueError(
                        f"particle_fill.container_collision_proxy.boxes[{box_index}].position_m "
                        "must contain three finite numbers"
                    )
                if len(box_size) != 3 or not all(
                    math.isfinite(value) and 0.0001 <= value <= 10.0 for value in box_size
                ):
                    raise ValueError(
                        f"particle_fill.container_collision_proxy.boxes[{box_index}].size_m "
                        "must contain three values from 0.0001 to 10 metres"
                    )
                boxes.append({"position_m": box_position, "size_m": box_size})
            proxy_friction = float(raw_proxy.get("friction", friction))
            if not math.isfinite(proxy_friction) or not 0.0 <= proxy_friction <= 10.0:
                raise ValueError(
                    "particle_fill.container_collision_proxy.friction must be between 0 and 10"
                )
            proxy = {
                "body_path": body_path,
                "source_shape_path": source_shape_path,
                "boxes": boxes,
                "friction": proxy_friction,
            }
        clean_particle_fill = {
            "name": name,
            "position_m": position,
            "dimensions": dimensions,
            "spacing_m": spacing,
            "radius_m": radius,
            "mass_kg": mass,
            "jitter_m": jitter,
            "friction": friction,
            "cohesion": cohesion,
            "adhesion": adhesion,
            "max_velocity_m_s": max_velocity,
            "color_rgb": color,
            "particle_count": particle_count,
            "container_collision_proxy": proxy,
        }
    return {
        "kind": "blacknode.newton-scene",
        "schema_version": 2,
        "asset_path": str(path),
        "source_asset": str(asset_path),
        "root_path": clean_root,
        "fixed_base": bool(fixed_base),
        "home_positions": clean_home,
        "rigid_bodies": clean_bodies,
        "particle_fill": clean_particle_fill,
        "ground": {"enabled": bool(ground_enabled), "height_m": clean_ground_height},
        "self_collisions": bool(self_collisions),
        "render": {"show_colliders": bool(show_colliders)},
        "collision": {
            "convex_decomposition_patterns": clean_patterns,
            "friction_overrides": clean_friction,
            "mass_overrides": (
                dict(DEFAULT_PICK_MASS_OVERRIDES) if default_pick_scene else {}
            ),
            "static_body_patterns": (
                list(DEFAULT_STATIC_COLLISION_BODY_PATTERNS)
                if default_pick_scene
                else []
            ),
            "mesh_approximation_overrides": (
                dict(DEFAULT_MESH_APPROXIMATION_OVERRIDES)
                if default_pick_scene
                else {}
            ),
            "grip_pads": (
                copy.deepcopy(list(DEFAULT_GRIP_PAD_SPECS))
                if default_pick_scene
                else []
            ),
            "compound_shape_proxies": (
                copy.deepcopy(list(DEFAULT_CONTAINER_COLLISION_PROXIES))
                if default_pick_scene
                else []
            ),
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
        "particle_fill": {},
        "ground": {"enabled": False, "height_m": 0.0},
        "self_collisions": False,
        "render": {"show_colliders": False},
        "collision": {
            "convex_decomposition_patterns": [],
            "friction_overrides": {},
        },
    }


def make_default_workspace_scene_spec() -> dict[str, Any]:
    """Load the bundled SO-101 tabletop scene used for quick workspace tests."""
    return make_usd_scene_spec(
        asset_path=package_asset_uri(DEFAULT_WORKSPACE_SCENE),
        root_path="/so101_new_calib",
        fixed_base=True,
        ground_enabled=False,
        ground_height=0.0,
        self_collisions=False,
        show_colliders=False,
        home_positions={},
        rigid_bodies=[],
        convex_decomposition_patterns=[],
        friction_overrides={},
        particle_fill={},
    )


def _default_workspace_editor_state() -> dict[str, Any]:
    return {
        "selected_path": "",
        "show_grid": False,
        "show_visuals": True,
        "show_colliders": False,
        "environment": {
            "background_color": "#6383c5",
            "hdri": "apartment",
            "hdri_path": "",
            "hdri_enabled": True,
            "show_background": True,
            "intensity": 1.0,
            "distant_light": {
                "enabled": True,
                "intensity": 2500.0,
                "color": "#fff2e0",
                "angle_deg": 4.0,
                "rotation_deg": [-35.0, 25.0, -25.0],
            },
        },
        "visibility": {},
        "transforms": {},
        "materials": {},
        "joint_drive_overrides": {
            "gripper": dict(DEFAULT_GRIPPER_DRIVE),
        },
        "joint_motion_limits": {},
    }


def _finite_vector(name: str, value: Any, *, positive: bool = False) -> list[float]:
    result = [float(item) for item in list(value or [])]
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite numbers")
    if positive and not all(item > 0.0 for item in result):
        raise ValueError(f"{name} values must be greater than zero")
    return result


def _clean_hex_color(value: Any) -> str:
    color = str(value or "").strip().lower()
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError("background_color must use #rrggbb format")
    try:
        int(color[1:], 16)
    except ValueError as exc:
        raise ValueError("background_color must use #rrggbb format") from exc
    return color


def _updated_workspace_environment(
    current: dict[str, Any], values: dict[str, Any]
) -> dict[str, Any]:
    environment = copy.deepcopy(current)
    if "background_color" in values:
        environment["background_color"] = _clean_hex_color(values["background_color"])
    if "hdri" in values:
        requested_hdri = str(values.get("hdri") or "none").strip().lower()
        if requested_hdri not in WORKSPACE_HDRI_PRESETS:
            raise ValueError(f"unknown HDRI preset: {requested_hdri}")
        environment["hdri"] = requested_hdri
        if requested_hdri != "custom":
            environment["hdri_path"] = ""
    if "hdri_path" in values:
        raw_path = str(values.get("hdri_path") or "").strip()
        if raw_path:
            hdri_path = Path(raw_path).expanduser().resolve()
            if not hdri_path.is_file() or hdri_path.suffix.lower() not in {".hdr", ".exr"}:
                raise ValueError("custom HDRI must be an existing .hdr or .exr file")
            environment["hdri_path"] = str(hdri_path)
            environment["hdri"] = "custom"
        else:
            environment["hdri_path"] = ""
            if environment.get("hdri") == "custom":
                environment["hdri"] = "none"
    if "show_background" in values:
        environment["show_background"] = bool(values["show_background"])
    if "hdri_enabled" in values:
        environment["hdri_enabled"] = bool(values["hdri_enabled"])
    if "intensity" in values:
        intensity = float(values["intensity"])
        if not math.isfinite(intensity) or not 0.0 <= intensity <= 100.0:
            raise ValueError("environment intensity must be between zero and 100")
        environment["intensity"] = intensity
    if "distant_light" in values:
        requested = dict(values.get("distant_light") or {})
        light = copy.deepcopy(environment.get("distant_light") or {})
        if "enabled" in requested:
            light["enabled"] = bool(requested["enabled"])
        if "intensity" in requested:
            intensity = float(requested["intensity"])
            if not math.isfinite(intensity) or not 0.0 <= intensity <= 1.0e7:
                raise ValueError("distant-light intensity must be between zero and 10000000")
            light["intensity"] = intensity
        if "color" in requested:
            light["color"] = _clean_hex_color(requested["color"])
        if "angle_deg" in requested:
            angle = float(requested["angle_deg"])
            if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
                raise ValueError("distant-light angle must be between zero and 180 degrees")
            light["angle_deg"] = angle
        if "rotation_deg" in requested:
            light["rotation_deg"] = _finite_vector(
                "distant-light rotation_deg", requested["rotation_deg"]
            )
        environment["distant_light"] = light
    return environment


def _workspace_material_path(path: str) -> str:
    token = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{WORKSPACE_MATERIAL_ROOT}/material_{token}"


def _workspace_material_keeper_path(path: str) -> str:
    token = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{WORKSPACE_MATERIAL_ROOT}/__PopulationKeepers/keeper_{token}"


def _collision_display_path(path: str) -> str:
    token = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return f"{COLLIDER_DISPLAY_ROOT}/Geometry/collider_{token}"


def _visual_display_path(path: str) -> str:
    token = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return f"{VISUAL_DISPLAY_ROOT}/Geometry/visual_{token}"


def _visual_display_geometry_path(path: str) -> str:
    # Kept as the semantic geometry accessor for tests and callers. Visual
    # display proxies are standalone Gprims because OVStage live transforms do
    # not propagate through referenced child hierarchies.
    return _visual_display_path(path)


def _usd_local_transform(
    prim: Any, UsdGeom: Any, Gf: Any, meters_per_unit: float
) -> dict[str, list[float]]:
    result = {
        "translate_m": [0.0, 0.0, 0.0],
        "rotate_deg": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }
    try:
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        transform = Gf.Transform(matrix)
        result["translate_m"] = [
            float(value) * meters_per_unit for value in transform.GetTranslation()
        ]
        result["scale"] = [float(value) for value in transform.GetScale()]
        result["rotate_deg"] = [
            float(value)
            for value in transform.GetRotation().Decompose(
                Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()
            )
        ]
    except Exception:
        pass
    return result


def _usd_rigid_matrix(matrix: Any, Gf: Any) -> Any:
    """Remove authored scale while retaining a body's world pose."""
    source = Gf.Transform(matrix)
    rigid = Gf.Transform()
    rigid.SetRotation(source.GetRotation())
    rigid.SetTranslation(source.GetTranslation())
    return rigid.GetMatrix()


def _editor_transform_matrix_m(transform: dict[str, Any], Gf: Any) -> Any:
    """Build a row-vector USD matrix whose translation is expressed in metres."""
    value = Gf.Transform()
    value.SetTranslation(Gf.Vec3d(*(
        float(item) for item in transform.get("translate_m", [0.0, 0.0, 0.0])
    )))
    rotate = [float(item) for item in transform.get("rotate_deg", [0.0, 0.0, 0.0])]
    value.SetRotation(
        Gf.Rotation(Gf.Vec3d.XAxis(), rotate[0])
        * Gf.Rotation(Gf.Vec3d.YAxis(), rotate[1])
        * Gf.Rotation(Gf.Vec3d.ZAxis(), rotate[2])
    )
    value.SetScale(Gf.Vec3d(*(
        float(item) for item in transform.get("scale", [1.0, 1.0, 1.0])
    )))
    return value.GetMatrix()


def _newton_pose_matrix_m(pose: list[float], Gf: Any) -> Any:
    """Convert Newton's ``xyz + xyzw`` pose to a row-vector USD matrix."""
    value = Gf.Transform()
    value.SetTranslation(Gf.Vec3d(*(float(item) for item in pose[:3])))
    value.SetRotation(Gf.Rotation(Gf.Quatd(
        float(pose[6]), Gf.Vec3d(*(float(item) for item in pose[3:6]))
    )))
    return value.GetMatrix()


def _newton_pose_from_matrix_m(matrix: Any, Gf: Any) -> list[float]:
    """Extract a scale-free Newton ``xyz + xyzw`` pose from a USD matrix."""
    rigid = Gf.Transform(_usd_rigid_matrix(matrix, Gf))
    quaternion = rigid.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return [
        *(float(item) for item in rigid.GetTranslation()),
        *(float(item) for item in imaginary),
        float(quaternion.GetReal()),
    ]


def _usd_path_has_collision_segment(path: Any) -> bool:
    return any(
        segment in {"collision", "collisions", "collider", "colliders"}
        for segment in str(path).strip("/").lower().split("/")
    )


def _is_collider_display_path(path: Any) -> bool:
    value = str(path)
    return value == COLLIDER_DISPLAY_ROOT or value.startswith(COLLIDER_DISPLAY_ROOT + "/")


def _is_internal_display_path(path: Any) -> bool:
    value = str(path)
    return (
        _is_collider_display_path(value)
        or value == VISUAL_DISPLAY_ROOT
        or value.startswith(VISUAL_DISPLAY_ROOT + "/")
        or value == WORKSPACE_MATERIAL_ROOT
        or value.startswith(WORKSPACE_MATERIAL_ROOT + "/")
    )


def _usd_is_collision_prim(prim: Any, UsdPhysics: Any) -> bool:
    cursor = prim
    while cursor and cursor.IsValid():
        if cursor.HasAPI(UsdPhysics.CollisionAPI) or _usd_path_has_collision_segment(
            cursor.GetPath()
        ):
            return True
        cursor = cursor.GetParent()
    return False


def _usd_is_collision_only_prim(prim: Any, UsdGeom: Any, UsdPhysics: Any) -> bool:
    if not _usd_is_collision_prim(prim, UsdPhysics):
        return False
    if _usd_path_has_collision_segment(prim.GetPath()):
        return True
    try:
        return UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide
    except Exception:
        return False


def _normalized_collision_import_status(
    normalized_paths: set[str],
    path_shape_map: dict[Any, Any],
    shape_flags: list[Any],
    collision_bit: int,
) -> tuple[set[int], list[str], list[str]]:
    """Resolve normalized USD colliders through Newton's authoritative import map."""
    imported_shapes = {str(path): shape_index for path, shape_index in path_shape_map.items()}
    active_shape_indices: set[int] = set()
    missing_paths: list[str] = []
    inactive_paths: list[str] = []
    for path in sorted(normalized_paths):
        # The OVRT collision overlay contains render-only internal references.
        # It is deliberately deactivated during Newton import and therefore
        # must never participate in the physics-collider safety audit.
        if _is_collider_display_path(path):
            continue
        raw_shape_index = imported_shapes.get(path)
        if raw_shape_index is None:
            missing_paths.append(path)
            continue
        try:
            shape_index = int(raw_shape_index)
        except (TypeError, ValueError):
            missing_paths.append(path)
            continue
        if shape_index < 0 or shape_index >= len(shape_flags):
            missing_paths.append(path)
            continue
        if int(shape_flags[shape_index]) & collision_bit:
            active_shape_indices.add(shape_index)
        else:
            inactive_paths.append(path)
    return active_shape_indices, missing_paths, inactive_paths


def _hide_generated_convex_collision_visuals(
    shape_labels: list[Any],
    shape_flags: list[Any],
    decomposed_source_paths: set[str],
    visible_bit: int,
) -> list[int]:
    """Keep importer-generated CoACD hulls physical while hiding their debug geometry."""
    hidden: list[int] = []
    prefixes = tuple(f"{path}_convex_" for path in sorted(decomposed_source_paths))
    if not prefixes:
        return hidden
    for shape_index, label in enumerate(shape_labels):
        if not str(label).startswith(prefixes):
            continue
        shape_flags[shape_index] = int(shape_flags[shape_index]) & ~int(visible_bit)
        hidden.append(shape_index)
    return hidden


def _connected_body_indices(
    seeds: set[int], joint_parents: list[Any], joint_children: list[Any]
) -> set[int]:
    """Return the complete rigid-body component containing controlled joints."""
    adjacency: dict[int, set[int]] = {}
    for raw_parent, raw_child in zip(joint_parents, joint_children):
        parent = int(raw_parent)
        child = int(raw_child)
        if child < 0:
            continue
        adjacency.setdefault(child, set())
        if parent >= 0:
            adjacency[child].add(parent)
            adjacency.setdefault(parent, set()).add(child)
    connected = {int(value) for value in seeds if int(value) >= 0}
    pending = list(connected)
    while pending:
        body = pending.pop()
        for neighbor in adjacency.get(body, set()):
            if neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)
    return connected


def _configure_mujoco_gravity_compensation(
    builder: Any,
    joint_drive_indices: dict[str, int],
    joint_child_bodies: dict[str, int],
    raw_config: Any,
) -> dict[str, Any]:
    """Enable MuJoCo gravity compensation for selected articulation joints.

    Body compensation is applied to the complete connected robot component,
    while free props remain outside that component and continue to feel
    gravity.  Joint actuator compensation is limited to the selected one-DOF
    drives so callers can leave a gripper drive uncompensated.
    """
    if isinstance(raw_config, dict):
        enabled = bool(raw_config.get("enabled", True))
        requested_names = [
            str(value) for value in list(raw_config.get("joint_names") or [])
        ]
        excluded_names = {
            str(value) for value in list(raw_config.get("exclude_joint_names") or [])
        }
        body_factor = float(raw_config.get("body_factor", 1.0))
        body_mode = str(raw_config.get("body_mode", "connected")).strip().lower()
    else:
        enabled = bool(raw_config)
        requested_names = []
        excluded_names = set()
        body_factor = 1.0
        body_mode = "connected"

    result = {
        "enabled": enabled,
        "joint_names": [],
        "body_indices": [],
        "body_factor": body_factor,
        "body_mode": body_mode,
    }
    if not enabled:
        return result
    if not math.isfinite(body_factor) or not 0.0 <= body_factor <= 1.0:
        raise ValueError(
            "mujoco_gravity_compensation.body_factor must be between 0 and 1"
        )
    if body_mode not in {"connected", "selected"}:
        raise ValueError(
            "mujoco_gravity_compensation.body_mode must be connected or selected"
        )

    available_names = set(joint_drive_indices)
    selected_names = (
        set(requested_names) if requested_names else available_names
    ) - excluded_names
    unknown_names = sorted((set(requested_names) | excluded_names) - available_names)
    if unknown_names:
        raise ValueError(
            "mujoco_gravity_compensation contains unknown one-DOF joints: "
            + ", ".join(unknown_names)
        )
    if not selected_names:
        raise ValueError(
            "mujoco_gravity_compensation must select at least one one-DOF joint"
        )

    actgravcomp_attr = builder.custom_attributes.get("mujoco:jnt_actgravcomp")
    body_gravcomp_attr = builder.custom_attributes.get("mujoco:gravcomp")
    if actgravcomp_attr is None or body_gravcomp_attr is None:
        raise RuntimeError(
            "MuJoCo gravity-compensation attributes were not registered"
        )
    if actgravcomp_attr.values is None:
        actgravcomp_attr.values = {}
    if body_gravcomp_attr.values is None:
        body_gravcomp_attr.values = {}

    for name in sorted(selected_names):
        actgravcomp_attr.values[int(joint_drive_indices[name])] = True

    body_seeds = {int(joint_child_bodies[name]) for name in selected_names}
    body_indices = (
        _connected_body_indices(
            body_seeds,
            list(builder.joint_parent),
            list(builder.joint_child),
        )
        if body_mode == "connected"
        else body_seeds
    )
    compensated_bodies = []
    for body_index in sorted(body_indices):
        if body_index < 0 or float(builder.body_mass[body_index]) <= 0.0:
            continue
        body_gravcomp_attr.values[body_index] = body_factor
        compensated_bodies.append(body_index)

    result["joint_names"] = sorted(selected_names)
    result["body_indices"] = compensated_bodies
    return result


def _apply_workspace_usd_edits(stage: Any, edits: dict[str, Any]) -> list[dict[str, Any]]:
    """Author non-destructive workspace overrides and return an outliner snapshot."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage.SetEditTarget(stage.GetRootLayer())
    meters_per_unit = max(1.0e-12, float(UsdGeom.GetStageMetersPerUnit(stage)))

    # Some DCC exports author material:binding without applying the schema.
    # Normalize that metadata in the disposable overlay so both OpenUSD and
    # OVStage resolve original and live-edited materials consistently.
    for prim in list(stage.Traverse()):
        if prim.HasRelationship("material:binding"):
            UsdShade.MaterialBindingAPI.Apply(prim)

    collision_palette = (
        (1.0, 0.18, 0.12), (0.12, 0.82, 1.0), (0.95, 0.78, 0.08),
        (0.72, 0.24, 1.0), (0.18, 0.95, 0.42), (1.0, 0.35, 0.72),
    )
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdGeom.Gprim) or not _usd_is_collision_only_prim(
            prim, UsdGeom, UsdPhysics
        ):
            continue
        color_index = int(hashlib.sha1(str(prim.GetPath()).encode("utf-8")).hexdigest()[:8], 16)
        color = collision_palette[color_index % len(collision_palette)]
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr([0.42])

    def authored_material_state(prim: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "base_color": [0.7, 0.7, 0.7],
            "metallic": 0.0,
            "roughness": 0.5,
            "opacity": 1.0,
        }
        if prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            display_color = gprim.GetDisplayColorAttr().Get()
            display_opacity = gprim.GetDisplayOpacityAttr().Get()
            if display_color:
                state["base_color"] = [float(value) for value in display_color[0]]
            if display_opacity:
                state["opacity"] = float(display_opacity[0])
        try:
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not bound:
                return state
            shader = None
            for context in ("", "mdl", "mtlx"):
                source = bound.ComputeSurfaceSource(context)[0]
                if source and source.GetPrim().IsValid():
                    shader = source
                    break
            if shader is None:
                return state
            mappings = {
                "base_color": ("diffuseColor", "diffuse_color_constant", "base_color"),
                "metallic": ("metallic", "metallic_constant", "metalness"),
                "roughness": ("roughness", "reflection_roughness_constant", "specular_roughness"),
                "opacity": ("opacity", "opacity_constant"),
            }
            for field, input_names in mappings.items():
                for input_name in input_names:
                    shader_input = shader.GetInput(input_name)
                    if not shader_input:
                        continue
                    value = shader_input.Get()
                    if value is None:
                        continue
                    if field == "base_color":
                        values = [float(item) for item in value]
                        if len(values) >= 3:
                            state[field] = values[:3]
                    else:
                        state[field] = float(value)
                    break
        except Exception:
            # Unusual renderer-specific networks still retain displayColor or
            # the portable preview defaults in the basic editor.
            pass
        return state

    source_materials = {
        str(prim.GetPath()): authored_material_state(prim)
        for prim in list(stage.Traverse())
        if prim.IsA(UsdGeom.Gprim)
    }
    for path, visible in dict(edits.get("visibility") or {}).items():
        prim = stage.GetPrimAtPath(str(path))
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if bool(visible) else UsdGeom.Tokens.invisible
            )
    for path, raw in dict(edits.get("transforms") or {}).items():
        prim = stage.GetPrimAtPath(str(path))
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            continue
        transform = Gf.Transform()
        translation_m = _finite_vector("translate_m", raw.get("translate_m"))
        transform.SetTranslation(Gf.Vec3d(*(value / meters_per_unit for value in translation_m)))
        rotation = (
            Gf.Rotation(Gf.Vec3d.XAxis(), float(raw["rotate_deg"][0]))
            * Gf.Rotation(Gf.Vec3d.YAxis(), float(raw["rotate_deg"][1]))
            * Gf.Rotation(Gf.Vec3d.ZAxis(), float(raw["rotate_deg"][2]))
        )
        transform.SetRotation(rotation)
        transform.SetScale(Gf.Vec3d(*_finite_vector("scale", raw.get("scale"), positive=True)))
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(transform.GetMatrix())
    for path, raw in dict(edits.get("materials") or {}).items():
        prim = stage.GetPrimAtPath(str(path))
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            continue
        material_path = _workspace_material_path(str(path))
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        color = _finite_vector("base_color", raw.get("base_color"))
        if not all(0.0 <= value <= 1.0 for value in color):
            raise ValueError("base_color values must be between zero and one")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        for name in ("metallic", "roughness", "opacity"):
            value = float(raw.get(name, {"metallic": 0.0, "roughness": 0.5, "opacity": 1.0}[name]))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    # Live OVStage material binding can only target materials retained during
    # renderer population. Keep every editable material referenced by an
    # invisible internal Gprim; otherwise an initially unbound material may be
    # pruned and the first edit resolves to the renderer's red error material.
    # The source Gprim keeps its authored material until the operator edits it.
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Gprim) or path in dict(edits.get("materials") or {}):
            continue
        material_path = _workspace_material_path(path)
        UsdShade.MaterialBindingAPI.Apply(prim)
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        source = source_materials.get(path, {})
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*source.get("base_color", [0.7, 0.7, 0.7]))
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(source.get("metallic", 0.0))
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(source.get("roughness", 0.5))
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            float(source.get("opacity", 1.0))
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        keeper = UsdGeom.Mesh.Define(stage, _workspace_material_keeper_path(path))
        keeper.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdShade.MaterialBindingAPI.Apply(keeper.GetPrim()).Bind(material)

    # Keep source visual geometry composed through the original sublayer. USD
    # scenes can contain hundreds of megabytes of mesh arrays; copying every
    # Gprim into the workspace overlay makes loading scale with duplicated
    # geometry and can discard hierarchy-specific behavior. OVRT live matrices
    # target the original render Gprims directly below.
    source_gprims = [
        prim
        for prim in list(stage.Traverse())
        if prim.IsA(UsdGeom.Gprim) and not _is_internal_display_path(prim.GetPath())
    ]
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # Collision references share source geometry but are surface-hidden by the
    # OVRT worker; their topology drives the green screen-space wireframe.
    collision_materials = UsdGeom.Scope.Define(stage, f"{COLLIDER_DISPLAY_ROOT}/Looks")
    del collision_materials
    for prim in source_gprims:
        if not _usd_is_collision_prim(prim, UsdPhysics):
            continue
        source_path = str(prim.GetPath())
        proxy_path = _collision_display_path(source_path)
        proxy = stage.DefinePrim(proxy_path)
        proxy.GetReferences().AddInternalReference(source_path)
        proxy.CreateAttribute(
            "blacknode:sourcePath", Sdf.ValueTypeNames.String
        ).Set(source_path)
        proxy.SetDisplayName(f"{prim.GetName()} collision")
        proxy_xform = UsdGeom.Xformable(proxy)
        proxy_xform.ClearXformOpOrder()
        proxy_xform.SetResetXformStack(True)
        proxy_xform.MakeMatrixXform().Set(xform_cache.GetLocalToWorldTransform(prim))
        proxy_imageable = UsdGeom.Imageable(proxy)
        # OVRT's rendering population intentionally excludes guide-purpose
        # prims. These display-only copies use default purpose while remaining
        # outside Newton's imported physics root.
        proxy_imageable.CreatePurposeAttr(UsdGeom.Tokens.default_)

        color_index = int(hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8], 16)
        color = (0.20, 1.0, 0.32)
        material_path = f"{COLLIDER_DISPLAY_ROOT}/Looks/material_{color_index:08x}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(proxy).Bind(material)

    transforms = dict(edits.get("transforms") or {})
    materials = dict(edits.get("materials") or {})
    items: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            path.startswith("/__BlacknodeMaterials")
            or path.startswith(COLLIDER_DISPLAY_ROOT)
            or path.startswith(VISUAL_DISPLAY_ROOT)
        ):
            continue
        imageable = prim.IsA(UsdGeom.Imageable)
        xformable = prim.IsA(UsdGeom.Xformable)
        if not imageable and not xformable:
            continue
        if prim.IsA(UsdGeom.Gprim):
            semantic_schemas = Sdf.TokenListOp()
            semantic_schemas.prependedItems = list(dict.fromkeys([
                *prim.GetAppliedSchemas(), "SemanticsAPI:class", "SemanticsAPI:label",
            ]))
            prim.SetMetadata("apiSchemas", semantic_schemas)
            semantic_class = (prim.GetTypeName() or "object").lower()
            semantic_label = prim.GetName() or path.rsplit("/", 1)[-1] or "object"
            for instance, value in (("class", semantic_class), ("label", semantic_label)):
                prim.CreateAttribute(
                    f"semantic:{instance}:params:semanticType", Sdf.ValueTypeNames.String
                ).Set(instance)
                prim.CreateAttribute(
                    f"semantic:{instance}:params:semanticData", Sdf.ValueTypeNames.String
                ).Set(value)
        inherited_visible = True
        if imageable:
            inherited_visible = (
                UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
            )
        material = dict(source_materials.get(path) or {
            "base_color": [0.7, 0.7, 0.7],
            "metallic": 0.0,
            "roughness": 0.5,
            "opacity": 1.0,
        })
        material.update(dict(materials.get(path) or {}))
        items.append({
            "path": path,
            "parent_path": str(prim.GetParent().GetPath()) if prim.GetParent() else "/",
            "name": prim.GetDisplayName() or prim.GetName(),
            "type_name": prim.GetTypeName() or "Prim",
            "render_role": (
                "collider" if _usd_is_collision_prim(prim, UsdPhysics) else "visual"
            ),
            "collision_only": _usd_is_collision_only_prim(prim, UsdGeom, UsdPhysics),
            "visible": inherited_visible,
            "editable": bool(xformable),
            "material_editable": bool(prim.IsA(UsdGeom.Gprim)),
            "transform": dict(
                transforms.get(path) or _usd_local_transform(prim, UsdGeom, Gf, meters_per_unit)
            ),
            "material": material,
            "material_path": _workspace_material_path(path) if prim.IsA(UsdGeom.Gprim) else "",
        })
    return items


def _compose_workspace_stage(asset_path: str, edits: dict[str, Any]) -> tuple[Any, str, list[dict[str, Any]]]:
    from pxr import Sdf, Usd, UsdGeom

    source_path = str(Path(asset_path).resolve()).replace("\\", "/")
    source_stage = Usd.Stage.Open(source_path)
    if source_stage is None:
        raise RuntimeError(f"OpenUSD could not open {asset_path}")
    root = Sdf.Layer.CreateAnonymous("blacknode-workspace.usda")
    root.subLayerPaths = [source_path]
    stage = Usd.Stage.Open(root)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not compose {asset_path}")
    # Stage-level metadata is not inherited from a sublayer. Preserve the
    # source coordinate system on the stronger workspace root so camera
    # framing and authored transforms remain in the original units.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    source_default_prim = source_stage.GetDefaultPrim()
    if source_default_prim and source_default_prim.IsValid():
        composed_default_prim = stage.GetPrimAtPath(source_default_prim.GetPath())
        if composed_default_prim and composed_default_prim.IsValid():
            stage.SetDefaultPrim(composed_default_prim)
    items = _apply_workspace_usd_edits(stage, edits)
    handle = tempfile.NamedTemporaryFile(prefix="blacknode-newton-", suffix=".usda", delete=False)
    handle.close()
    overlay_path = str(Path(handle.name).resolve())
    if not stage.GetRootLayer().Export(overlay_path):
        Path(overlay_path).unlink(missing_ok=True)
        raise RuntimeError("OpenUSD could not export the workspace overlay")
    return stage, overlay_path, items


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
        # Contact-sensitive manipulation needs to observe every 500 Hz physics
        # frame. Rendering remains independently rate-limited below.
        self.fps = max(10, min(500, int(fps)))
        self.substeps = max(1, min(MAX_PHYSICS_SUBSTEPS, int(substeps)))
        self.solver_iterations = max(1, min(64, int(solver_iterations)))
        self.joint_stiffness = self._validate_drive_gain("joint_stiffness", joint_stiffness)
        self.joint_damping = self._validate_drive_gain("joint_damping", joint_damping)
        self.joint_drive_overrides = self._validate_drive_overrides(joint_drive_overrides)
        self.joint_drive_gains: dict[str, dict[str, float]] = {}
        self.gravity_compensation: dict[str, Any] = {
            "enabled": False,
            "joint_names": [],
            "body_indices": [],
            "body_factor": 1.0,
            "body_mode": "connected",
        }
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
        self.external_joint_observation_source = ""
        self.external_joint_observation_received_at = 0.0
        self.external_joint_observation_observed_at = 0.0
        self.external_joint_observation_stale_after = 0.5
        self.external_joint_positions: dict[str, float] = {}
        self.stream_follow_source = ""
        self.stream_follow_authorized_source = ""
        self.stream_follow_deadline = 0.0
        self.stream_follow_stale_after = 0.5
        self.reference_state: Any = None
        self.articulation_body_indices: set[int] = set()
        self.digital_twin_ghost = {
            "visible": True,
            "placement": "beside",
            "offset_m": [0.35, 0.0, 0.0],
            "beside_offset_m": [0.35, 0.0, 0.0],
            "opacity": 0.28,
            "color_rgb": [0.18, 0.86, 1.0],
        }
        self.digital_twin_history: list[dict[str, Any]] = []
        self.digital_twin_baseline: dict[str, Any] = {}
        self.armed = False
        self.paused = False
        self.pending_steps = 0
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
        self.dynamic_body_indices: set[int] = set()
        self.body_pose_overrides: dict[int, list[float]] = {}
        self.free_joint_state_starts: dict[int, tuple[int, int]] = {}
        self.authored_dynamic_body_names: list[str] = []
        self.active_collision_shapes = 0
        self.usd_mesh_count = 0
        self.usd_meshes_with_normals = 0
        self.usd_collision_mesh_count = 0
        self.scene_items: list[dict[str, Any]] = []
        self.render_shapes: list[dict[str, Any]] = []
        self.show_visuals = bool(self.viewer_config.get("show_visuals", True))
        self.show_colliders = bool(
            self.viewer_config.get(
                "show_colliders", dict(self.scene.get("render") or {}).get("show_colliders", False)
            )
        )
        self.render_asset_path = str(self.scene.get("asset_path") or "")
        self._workspace_overlay_path = ""
        self._generated_render_path = ""
        self.joint_indices: dict[str, int] = {}
        self.joint_drive_indices: dict[str, int] = {}
        self.joint_limits: dict[str, tuple[float, float]] = {}
        self.joint_units: dict[str, str] = {}
        self.joint_dynamics: dict[str, dict[str, Any]] = {}
        self.joint_motion_limits: dict[str, dict[str, float]] = {}
        self.friction_override_matches: dict[str, list[int]] = {}
        self.grip_pad_shape_indices: list[int] = []
        self.grip_pad_source_shape_indices: list[int] = []
        self.grip_pad_bindings: list[tuple[int, list[float]]] = []
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

    def _export_model_render_asset(self, newton: Any) -> None:
        """Export frame zero for non-USD assets so every viewer shares one model."""
        from pxr import Usd

        handle = tempfile.NamedTemporaryFile(
            prefix="blacknode-newton-model-", suffix=".usda", delete=False
        )
        path = Path(handle.name)
        handle.close()
        path.unlink(missing_ok=True)
        viewer = newton.viewer.ViewerUSD(
            str(path), fps=self.fps, up_axis="Z", num_frames=1
        )
        generated_shape_paths: dict[int, str] = {}
        try:
            viewer.show_visual = True
            # Export collider prims for inspection and OVRTX's wireframe
            # overlay, but author them hidden in the base layer. Some render
            # backends do not reliably apply a first-frame visibility override
            # to referenced generated shapes; visible collider surfaces then
            # obscure the imported MJCF/URDF visual meshes with debug colors.
            viewer.show_collision = False
            viewer.set_model(self.model)
            # ViewerUSD batches matching geometries. Its shape_N names identify
            # render batches, while instance_N identifies the original model
            # shape inside that batch. Preserve this authoritative mapping;
            # model shape indices are not USD batch indices.
            for batch in viewer._shape_instances.values():
                batch_path = str(viewer._get_path(batch.name))
                for instance_index, shape_index in enumerate(batch.model_shapes):
                    generated_shape_paths[int(shape_index)] = (
                        f"{batch_path}/instance_{instance_index}"
                    )
            viewer.begin_frame(0.0)
            viewer.log_state(self.state_0)
            viewer.end_frame()
        finally:
            viewer.close()
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            path.unlink(missing_ok=True)
            raise RuntimeError("Newton could not export a render layer for the robot description")
        shape_bodies = self.model.shape_body.numpy().tolist()
        for entry in self.render_shapes:
            shape_index = int(entry.get("shape_index", -1))
            if shape_index < 0 or shape_index >= len(shape_bodies):
                continue
            render_path = generated_shape_paths.get(shape_index)
            if not render_path:
                continue
            entry["path"] = render_path
            entry["body_index"] = int(shape_bodies[shape_index])
            body_index = int(shape_bodies[shape_index])
            if int(body_index) < 0:
                continue
            prim = stage.GetPrimAtPath(render_path)
            if prim and prim.IsValid():
                body_label = str(self.model.body_label[int(body_index)] or f"body_{body_index}")
                shape_label = str(self.model.shape_label[shape_index] or f"shape_{shape_index}")
                prim.SetDisplayName(f"{body_label} · {shape_label}")
        stage.GetRootLayer().Save()
        self._generated_render_path = str(path)
        _stage, self._workspace_overlay_path, self.scene_items = _compose_workspace_stage(
            str(path), dict(self.scene.get("workspace_edits") or {})
        )
        self.render_asset_path = self._workspace_overlay_path

    def _build(self) -> None:
        newton, wp = self._imports()
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        except Exception as exc:  # pragma: no cover - declared importer dependency
            raise RuntimeError("OpenUSD Python schemas are required to load USD scenes") from exc
        device = self._resolve_device(wp)
        collision_config = dict(self.scene.get("collision") or {})
        decomposition_patterns = [
            str(value) for value in collision_config.get("convex_decomposition_patterns") or []
        ]
        static_body_patterns = [
            str(value) for value in collision_config.get("static_body_patterns") or []
        ]
        mesh_approximation_overrides = {
            str(pattern): str(approximation)
            for pattern, approximation in dict(
                collision_config.get("mesh_approximation_overrides") or {}
            ).items()
        }
        sdf_max_resolution_overrides = {
            str(pattern): int(resolution)
            for pattern, resolution in dict(
                collision_config.get("sdf_max_resolution_overrides") or {}
            ).items()
        }
        mass_overrides = {
            str(pattern): float(mass)
            for pattern, mass in dict(collision_config.get("mass_overrides") or {}).items()
        }
        grip_pad_specs = [
            dict(value) for value in list(collision_config.get("grip_pads") or [])
        ]
        compound_shape_proxies = [
            dict(value)
            for value in list(collision_config.get("compound_shape_proxies") or [])
        ]
        root_path = str(self.scene.get("root_path") or "/").rstrip("/") or "/"
        normalized_collision_paths: set[str] = set()
        collision_only_paths: set[str] = set()
        builder = newton.ModelBuilder()
        if str(self.scene.get("solver") or "xpbd").strip().lower() == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        # Newton 1.5 changed the inherited rigid contact gap to 0.1 m. That
        # makes ordinary dynamic pairs enter MuJoCo constraints while they are
        # still up to 20 cm apart, which is catastrophic for tabletop scenes.
        # Real contact is the runtime default; callers may opt into a small
        # positive detection envelope explicitly when they actually need one.
        builder.rigid_gap = float(self.scene.get("rigid_contact_gap_m", 0.0))
        usd_import_result: dict[str, Any] = {}
        collision_proxy_shape_indices: set[int] = set()
        particle_fill = dict(self.scene.get("particle_fill") or {})
        container_collision_proxy = dict(
            particle_fill.get("container_collision_proxy") or {}
        )
        if container_collision_proxy:
            source_shape_path = str(
                container_collision_proxy.get("source_shape_path") or ""
            )
            compound_shape_proxies = [
                value
                for value in compound_shape_proxies
                if str(value.get("source_shape_path") or "") != source_shape_path
            ]
            compound_shape_proxies.append(container_collision_proxy)
        stage = None
        asset_path = str(self.scene.get("asset_path") or "").strip()
        asset_format = str(self.scene.get("asset_format") or "usd").strip().lower()

        def collision_authored(prim: Any) -> bool:
            return _usd_is_collision_prim(prim, UsdPhysics)

        collision_meshes = 0
        if asset_path and asset_format == "usd":
            workspace_edits = dict(self.scene.get("workspace_edits") or {})
            stage, self._workspace_overlay_path, self.scene_items = _compose_workspace_stage(
                asset_path, workspace_edits
            )
            self.render_asset_path = self._workspace_overlay_path
            # Some USD authoring tools place collision APIs on an Xform above
            # the mesh. Newton consumes them on the mesh itself. Normalize only
            # the in-memory stage so the source file stays unchanged.
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if _is_internal_display_path(path):
                    continue
                if root_path != "/" and path != root_path and not path.startswith(root_path + "/"):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    if any(pattern in path for pattern in static_body_patterns):
                        # Exact triangle meshes are stable environment
                        # colliders. Removing the body API keeps an open bin
                        # open instead of asking the solver to move a concave
                        # dynamic body.
                        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    for pattern, mass in mass_overrides.items():
                        if pattern in path:
                            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
                            break
                if prim.IsA(UsdGeom.Mesh):
                    self.usd_mesh_count += 1
                    normals = UsdGeom.Mesh(prim).GetNormalsAttr().Get()
                    if normals is not None and len(normals) > 0:
                        self.usd_meshes_with_normals += 1
                    if collision_authored(prim):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        approximation = next(
                            (
                                value
                                for pattern, value in mesh_approximation_overrides.items()
                                if pattern in path
                            ),
                            (
                                "convexDecomposition"
                                if any(pattern in path for pattern in decomposition_patterns)
                                else "convexHull"
                            ),
                        )
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
                        sdf_resolution = next(
                            (
                                value
                                for pattern, value in sdf_max_resolution_overrides.items()
                                if pattern in path
                            ),
                            None,
                        )
                        if sdf_resolution is not None:
                            if sdf_resolution <= 0 or sdf_resolution % 8 != 0:
                                raise ValueError(
                                    "collision.sdf_max_resolution_overrides values must be "
                                    f"positive multiples of 8; got {sdf_resolution} for {path}"
                                )
                            prim.CreateAttribute(
                                "newton:sdfMaxResolution",
                                Sdf.ValueTypeNames.Int,
                                custom=True,
                            ).Set(sdf_resolution)
                        collision_meshes += 1
                        normalized_collision_paths.add(path)
                if prim.IsA(UsdGeom.Gprim) and _usd_is_collision_only_prim(
                    prim, UsdGeom, UsdPhysics
                ):
                    collision_only_paths.add(path)
                if prim.HasRelationship("material:binding"):
                    UsdShade.MaterialBindingAPI.Apply(prim)
            self.usd_collision_mesh_count = collision_meshes
            # Preserve concave geometry with a coarse, deterministic
            # decomposition when the operator explicitly requests it.
            builder.default_mesh_approximation_cfg.coacd_threshold = 0.2
            display_root = stage.GetPrimAtPath(COLLIDER_DISPLAY_ROOT)
            visual_display_root = stage.GetPrimAtPath(VISUAL_DISPLAY_ROOT)
            material_root = stage.GetPrimAtPath(WORKSPACE_MATERIAL_ROOT)
            if display_root and display_root.IsValid():
                display_root.SetActive(False)
            if visual_display_root and visual_display_root.IsValid():
                visual_display_root.SetActive(False)
            if material_root and material_root.IsValid():
                material_root.SetActive(False)
            try:
                usd_import_result = builder.add_usd(
                    stage,
                    root_path=str(self.scene.get("root_path") or "/"),
                    # None preserves authored fixed/free joints. Passing False
                    # fixes every rigid body in the imported stage, including props.
                    floating=None if bool(self.scene.get("fixed_base", True)) else True,
                    enable_self_collisions=bool(self.scene.get("self_collisions", False)),
                    load_visual_shapes=True,
                    hide_collision_shapes=True,
                    force_show_colliders=False,
                    force_position_velocity_actuation=True,
                )
                if bool(collision_config.get("split_render_collision_meshes", False)):
                    # Newton normally shares one indexed Mesh between rendering
                    # and collision when a USD Gprim has both roles. Collision
                    # import intentionally welds face-varying vertices, which
                    # discards authored hard normals and visually rounds cubes,
                    # tables, and other sharp assets. Keep that original mesh
                    # collision-only and attach an authored-normal render-only
                    # copy to the same body and local transform.
                    visible_bit = int(newton.ShapeFlags.VISIBLE)
                    collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(
                        newton.ShapeFlags.COLLIDE_PARTICLES
                    )
                    original_shape_count = len(builder.shape_label)
                    for shape_index in range(original_shape_count):
                        flags = int(builder.shape_flags[shape_index])
                        if not (flags & visible_bit) or not (flags & collision_bits):
                            continue
                        if int(builder.shape_type[shape_index]) != int(newton.GeoType.MESH):
                            continue
                        source = builder.shape_source[shape_index]
                        if not isinstance(source, newton.Mesh):
                            continue
                        label = str(builder.shape_label[shape_index])
                        prim = stage.GetPrimAtPath(label)
                        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                            continue
                        authored = newton.usd.get_mesh(
                            prim,
                            load_normals=True,
                            load_uvs=source.texture is not None,
                            preserve_facevarying_uvs=source.texture is not None,
                        )
                        if authored.normals is None:
                            continue
                        render_mesh = newton.Mesh(
                            authored.vertices,
                            authored.indices,
                            normals=authored.normals,
                            uvs=authored.uvs,
                            compute_inertia=False,
                            is_solid=source.is_solid,
                            maxhullvert=source.maxhullvert,
                            color=source.color,
                            roughness=source.roughness,
                            metallic=source.metallic,
                            texture=source.texture,
                        )
                        render_cfg = newton.ModelBuilder.ShapeConfig(
                            density=0.0,
                            is_solid=False,
                            has_shape_collision=False,
                            has_particle_collision=False,
                            is_visible=True,
                        )
                        builder.add_shape_mesh(
                            body=int(builder.shape_body[shape_index]),
                            xform=builder.shape_transform[shape_index],
                            mesh=render_mesh,
                            scale=builder.shape_scale[shape_index],
                            cfg=render_cfg,
                            color=builder.shape_color[shape_index],
                            label=f"{label}_authored_render",
                        )
                        builder.shape_flags[shape_index] = flags & ~visible_bit
            finally:
                if display_root and display_root.IsValid():
                    display_root.SetActive(True)
                if visual_display_root and visual_display_root.IsValid():
                    visual_display_root.SetActive(True)
                if material_root and material_root.IsValid():
                    material_root.SetActive(True)
            _hide_generated_convex_collision_visuals(
                builder.shape_label,
                builder.shape_flags,
                {
                    path
                    for path in normalized_collision_paths
                    if any(pattern in path for pattern in decomposition_patterns)
                },
                int(newton.ShapeFlags.VISIBLE),
            )
        elif asset_path and asset_format == "urdf":
            source = str(self.scene.get("robot_description_xml") or asset_path)
            builder.add_urdf(
                source,
                floating=False if bool(self.scene.get("fixed_base", True)) else True,
                enable_self_collisions=bool(self.scene.get("self_collisions", False)),
                force_show_colliders=False,
                force_position_velocity_actuation=True,
            )
        elif asset_path and asset_format == "mjcf":
            fixed_base = self.scene.get("fixed_base")
            builder.add_mjcf(
                asset_path,
                floating=None if fixed_base is None else not bool(fixed_base),
                enable_self_collisions=bool(self.scene.get("self_collisions", False)),
                force_show_colliders=False,
                parse_visuals=True,
                parse_meshes=True,
            )
            mjcf_home_qpos = list(self.scene.get("mjcf_home_qpos") or [])
            if mjcf_home_qpos:
                _apply_mjcf_keyframe_qpos(builder, newton, mjcf_home_qpos)
        disabled_shape_patterns = [
            str(value)
            for value in list(collision_config.get("disable_shape_patterns") or [])
            if str(value)
        ]
        if disabled_shape_patterns:
            collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(
                newton.ShapeFlags.COLLIDE_PARTICLES
            )
            for shape_index, label in enumerate(builder.shape_label):
                if any(pattern in str(label) for pattern in disabled_shape_patterns):
                    builder.shape_flags[shape_index] = (
                        int(builder.shape_flags[shape_index]) & ~collision_bits
                    )
        for pad in grip_pad_specs:
            body_path = str(pad["body_path"])
            try:
                body_index = next(
                    index
                    for index, label in enumerate(builder.body_label)
                    if str(label) == body_path
                )
            except StopIteration as exc:
                raise RuntimeError(
                    f"grip pad body was not imported: {body_path}"
                ) from exc
            source_pattern = str(pad["source_shape_pattern"])
            source_shape_indices = [
                index
                for index, label in enumerate(builder.shape_label)
                if source_pattern in str(label)
            ]
            if not source_shape_indices:
                raise RuntimeError(
                    f"grip pad source shape was not imported: {source_pattern}"
                )
            collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(
                newton.ShapeFlags.COLLIDE_PARTICLES
            )
            for shape_index in source_shape_indices:
                builder.shape_flags[shape_index] = (
                    int(builder.shape_flags[shape_index]) & ~collision_bits
                )
            self.grip_pad_source_shape_indices.extend(source_shape_indices)
            position = [float(value) for value in pad["position_m"]]
            rotation = [float(value) for value in pad["rotation_xyzw"]]
            size = [float(value) for value in pad["size_m"]]
            pad_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=float(pad["friction"]),
                restitution=0.0,
                mu_torsional=float(pad.get("torsional_friction_m", 0.006)),
                mu_rolling=float(pad.get("rolling_friction_m", 0.001)),
                ke=float(pad.get("contact_stiffness_n_m", 2500.0)),
                kd=float(pad.get("contact_damping_n_s_m", 100.0)),
                margin=float(pad["margin_m"]),
                is_visible=False,
            )
            pad_shape_index = builder.add_shape_box(
                body_index,
                xform=wp.transform(position, wp.quat(*rotation)),
                hx=size[0] / 2.0,
                hy=size[1] / 2.0,
                hz=size[2] / 2.0,
                cfg=pad_cfg,
                label=str(pad["label"]),
            )
            self.grip_pad_shape_indices.append(pad_shape_index)
            self.grip_pad_bindings.append((body_index, [*position, *rotation]))
            replacement_paths = [
                path for path in normalized_collision_paths if source_pattern in path
            ]
            if not replacement_paths:
                raise RuntimeError(
                    f"grip pad source collision path was not normalized: {source_pattern}"
                )
            for path in replacement_paths:
                usd_import_result.setdefault("path_shape_map", {})[path] = pad_shape_index
            collision_proxy_shape_indices.add(pad_shape_index)
        if len(self.grip_pad_shape_indices) == 2:
            # The imported articulation disables self-collision globally. Opt
            # this one asset-specific pair back in so the opposing physical
            # fingertip colliders cannot cross when the gripper is empty.
            pad_pair = tuple(sorted(self.grip_pad_shape_indices))
            builder.shape_collision_filter_pairs = [
                pair
                for pair in builder.shape_collision_filter_pairs
                if tuple(sorted((int(pair[0]), int(pair[1])))) != pad_pair
            ]
        for compound_proxy in compound_shape_proxies:
            body_path = str(compound_proxy["body_path"])
            source_shape_path = str(compound_proxy["source_shape_path"])
            try:
                body_index = next(
                    index
                    for index, label in enumerate(builder.body_label)
                    if str(label) == body_path
                )
            except StopIteration as exc:
                raise RuntimeError(
                    f"container collision proxy body was not imported: {body_path}"
                ) from exc
            source_shape_indices = [
                index
                for index, label in enumerate(builder.shape_label)
                if str(label) == source_shape_path
                or str(label).startswith(source_shape_path + "_convex_")
            ]
            if not source_shape_indices:
                raise RuntimeError(
                    f"container collision proxy source shape was not imported: {source_shape_path}"
                )
            collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(
                newton.ShapeFlags.COLLIDE_PARTICLES
            )
            for shape_index in source_shape_indices:
                builder.shape_flags[shape_index] = (
                    int(builder.shape_flags[shape_index]) & ~collision_bits
                )
            source_collision_groups = {
                int(builder.shape_collision_group[shape_index])
                for shape_index in source_shape_indices
            }
            if len(source_collision_groups) != 1:
                raise RuntimeError(
                    f"collision proxy source shapes use different collision groups: "
                    f"{source_shape_path} -> {sorted(source_collision_groups)}"
                )
            source_collision_group = source_collision_groups.pop()
            proxy_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=float(compound_proxy["friction"]),
                restitution=0.0,
                mu_torsional=0.05,
                mu_rolling=0.01,
                # A replacement collider must preserve the source shape's
                # collision graph. Assigning the builder default creates a new
                # group (and can alias MuJoCo's finite contype bitmask), making
                # the proxy silently stop colliding with a gripper pad.
                collision_group=source_collision_group,
                sdf_max_resolution=(
                    int(compound_proxy["sdf_max_resolution"])
                    if compound_proxy.get("sdf_max_resolution") is not None
                    else None
                ),
                is_visible=False,
            )
            proxy_shape_indices: list[int] = []
            for box_index, box in enumerate(compound_proxy["boxes"]):
                size = [float(value) for value in box["size_m"]]
                position = [float(value) for value in box["position_m"]]
                proxy_shape_indices.append(
                    builder.add_shape_box(
                        body_index,
                        xform=wp.transform(position, wp.quat_identity()),
                        hx=size[0] / 2.0,
                        hy=size[1] / 2.0,
                        hz=size[2] / 2.0,
                        cfg=proxy_cfg,
                        label=f"{source_shape_path}_proxy_{box_index}",
                    )
                )
            collision_proxy_shape_indices.update(proxy_shape_indices)
            usd_import_result.setdefault("path_shape_map", {})[
                source_shape_path
            ] = proxy_shape_indices[0]
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
            self.dynamic_body_indices.add(body_index)
            self.authored_dynamic_body_names.append(name)
        friction_overrides = dict(collision_config.get("friction_overrides") or {})
        stiffness_overrides = dict(collision_config.get("contact_stiffness_overrides") or {})
        damping_overrides = dict(collision_config.get("contact_damping_overrides") or {})
        for shape_index, label in enumerate(builder.shape_label):
            text_label = str(label)
            for pattern, mu in friction_overrides.items():
                if str(pattern) in text_label:
                    builder.shape_material_mu[shape_index] = float(mu)
                    builder.shape_material_mu_torsional[shape_index] = 0.05
                    builder.shape_material_mu_rolling[shape_index] = 0.01
                    self.friction_override_matches.setdefault(str(pattern), []).append(shape_index)
                    break
            for pattern, stiffness in stiffness_overrides.items():
                if str(pattern) in text_label:
                    builder.shape_material_ke[shape_index] = float(stiffness)
                    break
            for pattern, damping in damping_overrides.items():
                if str(pattern) in text_label:
                    builder.shape_material_kd[shape_index] = float(damping)
                    break
        effort_overrides = {
            str(name): float(value)
            for name, value in dict(self.scene.get("joint_effort_overrides") or {}).items()
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in effort_overrides.values()):
            raise ValueError("joint_effort_overrides must contain positive finite torque/force limits")
        limit_overrides = {
            str(name): [float(bound) for bound in list(value)]
            for name, value in dict(self.scene.get("joint_limit_overrides") or {}).items()
        }
        for name, bounds in limit_overrides.items():
            if (
                len(bounds) != 2
                or not all(math.isfinite(bound) for bound in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(
                    f"joint_limit_overrides[{name!r}] must be finite [lower, upper]"
                )
        labels = list(builder.joint_label)
        starts = list(builder.joint_q_start)
        drive_starts = list(builder.joint_qd_start)
        home_positions = dict(self.scene.get("home_positions") or self.scene.get("home_radians") or {})
        for joint_id, (label, joint_type) in enumerate(zip(labels, builder.joint_type)):
            if int(joint_type) not in {int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)}:
                continue
            name = str(label).rsplit("/", 1)[-1]
            if not name or name in self.joint_indices:
                raise RuntimeError(
                    f"Newton teleoperation requires unique one-DOF joint names; duplicate: {name!r}"
                )
            index = int(starts[joint_id])
            drive_index = int(drive_starts[joint_id])
            self.joint_indices[name] = index
            self.joint_drive_indices[name] = drive_index
            if name in limit_overrides:
                builder.joint_limit_lower[drive_index] = limit_overrides[name][0]
                builder.joint_limit_upper[drive_index] = limit_overrides[name][1]
            lower = float(builder.joint_limit_lower[drive_index])
            upper = float(builder.joint_limit_upper[drive_index])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise RuntimeError(
                    f"joint {name!r} needs finite ordered limits for safe teleoperation, got [{lower}, {upper}]"
                )
            self.joint_limits[name] = (lower, upper)
            self.joint_units[name] = (
                "radians" if int(joint_type) == int(newton.JointType.REVOLUTE) else "metres"
            )
            self.joint_dynamics[name] = {
                "child_body_index": int(builder.joint_child[joint_id]),
                "passive_damping": float(builder.joint_damping[drive_index]),
            }
            self.joint_motion_limits[name] = {
                "max_velocity": self.max_velocity_rad_s,
                "max_step": self.max_step_rad,
            }
            initial = min(upper, max(lower, float(home_positions.get(name, builder.joint_q[index]))))
            builder.joint_q[index] = initial
            builder.joint_target_q[drive_index] = initial
            override = self.joint_drive_overrides.get(name, {})
            stiffness = float(override.get("stiffness", self.joint_stiffness))
            damping = float(override.get("damping", self.joint_damping))
            builder.joint_target_mode[drive_index] = int(
                newton.JointTargetMode.POSITION_VELOCITY
            )
            builder.joint_target_ke[drive_index] = stiffness
            builder.joint_target_kd[drive_index] = damping
            if name in effort_overrides:
                builder.joint_effort_limit[drive_index] = effort_overrides[name]
            self.joint_drive_gains[name] = {"stiffness": stiffness, "damping": damping}
            self.current[name] = initial
            self.desired[name] = initial
            self.applied[name] = initial
            self.home[name] = initial
        unknown_home = sorted(set(home_positions) - set(self.joint_indices))
        if unknown_home:
            raise RuntimeError("home_positions contains unknown one-DOF joints: " + ", ".join(unknown_home))
        unknown_limits = sorted(set(limit_overrides) - set(self.joint_indices))
        if unknown_limits:
            raise RuntimeError(
                "joint_limit_overrides contains unknown one-DOF joints: "
                + ", ".join(unknown_limits)
            )
        unknown_drives = sorted(set(self.joint_drive_overrides) - set(self.joint_indices))
        if unknown_drives:
            raise RuntimeError(
                "joint_drive_overrides contains unknown one-DOF joints: " + ", ".join(unknown_drives)
            )
        unknown_effort_limits = sorted(set(effort_overrides) - set(self.joint_indices))
        if unknown_effort_limits:
            raise RuntimeError(
                "joint_effort_overrides contains unknown one-DOF joints: "
                + ", ".join(unknown_effort_limits)
            )

        raw_gravity_compensation = self.scene.get("mujoco_gravity_compensation", False)
        if raw_gravity_compensation:
            if str(self.scene.get("solver") or "xpbd").strip().lower() != "mujoco":
                raise ValueError(
                    "mujoco_gravity_compensation requires solver='mujoco'"
                )
            self.gravity_compensation = _configure_mujoco_gravity_compensation(
                builder,
                self.joint_drive_indices,
                {
                    name: int(settings["child_body_index"])
                    for name, settings in self.joint_dynamics.items()
                },
                raw_gravity_compensation,
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
            self.dynamic_body_indices.add(body_index)
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
        if particle_fill:
            dimensions = [int(value) for value in particle_fill["dimensions"]]
            spacing = [float(value) for value in particle_fill["spacing_m"]]
            builder.add_particle_grid(
                pos=wp.vec3(*[float(value) for value in particle_fill["position_m"]]),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dimensions[0],
                dim_y=dimensions[1],
                dim_z=dimensions[2],
                cell_x=spacing[0],
                cell_y=spacing[1],
                cell_z=spacing[2],
                mass=float(particle_fill["mass_kg"]),
                jitter=float(particle_fill["jitter_m"]),
                radius_mean=float(particle_fill["radius_m"]),
            )
        ground = dict(self.scene.get("ground") or {})
        if ground.get("enabled", "ground_height_m" in self.scene):
            builder.add_ground_plane(
                height=float(ground.get("height_m", self.scene.get("ground_height_m") or 0.0))
            )
        # Newton's automatic estimate can be too small for an imported robot:
        # broadphase candidates from the articulated links can then consume the
        # whole allocation and silently drop the gripper/prop contacts.  Allow
        # contact-critical scenes to provide an explicit per-world budget.
        requested_contact_budget = int(self.scene.get("rigid_contact_max_per_world") or 0)
        if requested_contact_budget:
            builder.num_rigid_contacts_per_world = max(
                64, min(65_536, requested_contact_budget)
            )
        self.free_joint_state_starts = {
            int(child): (
                int(builder.joint_q_start[joint]),
                int(builder.joint_qd_start[joint]),
            )
            for joint, (child, joint_type) in enumerate(
                zip(builder.joint_child, builder.joint_type)
            )
            if int(joint_type) == int(newton.JointType.FREE)
        }
        self.model = builder.finalize(device=device)
        raw_contact_solref = self.scene.get("mujoco_contact_solref")
        if raw_contact_solref is not None:
            values = [float(value) for value in list(raw_contact_solref)]
            if (
                len(values) != 2
                or not all(math.isfinite(value) and value > 0.0 for value in values)
            ):
                raise ValueError(
                    "mujoco_contact_solref must be [positive time constant, "
                    "positive damping ratio]"
                )
            mujoco_attributes = getattr(self.model, "mujoco", None)
            solref_array = getattr(mujoco_attributes, "solref", None)
            solref_mode_array = getattr(mujoco_attributes, "solref_mode", None)
            if solref_array is None or solref_mode_array is None:
                raise RuntimeError(
                    "this Newton build does not expose MuJoCo per-shape solref attributes"
                )
            solref = solref_array.numpy()
            solref[:] = values
            solref_array.assign(solref)
            # Newton SolverMuJoCo constant: preserve the supplied values as
            # native MuJoCo (timeconst, dampratio), rather than converting the
            # generic force-space ke/kd fields again.
            solref_mode = solref_mode_array.numpy()
            solref_mode[:] = 1  # SOLREF_MODE_RAW
            solref_mode_array.assign(solref_mode)
        raw_grip_condim = self.scene.get("mujoco_grip_condim")
        if raw_grip_condim is not None:
            grip_condim = int(raw_grip_condim)
            if grip_condim not in {1, 3, 4, 6}:
                raise ValueError("mujoco_grip_condim must be one of 1, 3, 4, or 6")
            mujoco_attributes = getattr(self.model, "mujoco", None)
            condim_array = getattr(mujoco_attributes, "condim", None)
            if condim_array is None:
                raise RuntimeError(
                    "this Newton build does not expose MuJoCo per-shape condim attributes"
                )
            condim = condim_array.numpy()
            for shape_index in self.grip_pad_shape_indices:
                condim[int(shape_index)] = grip_condim
            condim_array.assign(condim)
        if particle_fill:
            self.model.particle_mu = float(particle_fill["friction"])
            self.model.particle_cohesion = float(particle_fill["cohesion"])
            self.model.particle_adhesion = float(particle_fill["adhesion"])
            self.model.particle_max_velocity = float(particle_fill["max_velocity_m_s"])
        try:
            controlled_bodies = {
                int(value["child_body_index"])
                for value in self.joint_dynamics.values()
            }
            self.articulation_body_indices = _connected_body_indices(
                controlled_bodies,
                self.model.joint_parent.numpy().tolist(),
                self.model.joint_child.numpy().tolist(),
            )
        except Exception:
            self.articulation_body_indices = set()
        shape_flags = self.model.shape_flags.numpy().tolist()
        shape_bodies = self.model.shape_body.numpy().tolist()
        visible_bit = int(newton.ShapeFlags.VISIBLE)
        collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        render_shapes_by_path: dict[str, dict[str, Any]] = {}
        proxy_sources: set[str] = set()
        visual_proxy_sources: set[str] = set()
        if stage is not None:
            render_xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            body_paths = sorted(
                (
                    (str(label), index)
                    for index, label in enumerate(self.model.body_label)
                    if str(label).startswith("/")
                    and stage.GetPrimAtPath(str(label)).IsValid()
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            body_bind_worlds = {
                index: _usd_rigid_matrix(
                    render_xform_cache.GetLocalToWorldTransform(
                        stage.GetPrimAtPath(path)
                    ),
                    Gf,
                )
                for path, index in body_paths
            }
            proxy_sources = {
                str(prim.GetAttribute("blacknode:sourcePath").Get() or "")
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(COLLIDER_DISPLAY_ROOT + "/")
                and prim.HasAttribute("blacknode:sourcePath")
            }
            visual_proxy_sources = {
                str(prim.GetAttribute("blacknode:sourcePath").Get() or "")
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(VISUAL_DISPLAY_ROOT + "/")
                and prim.HasAttribute("blacknode:sourcePath")
            }
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                collision_display_proxy = path.startswith(COLLIDER_DISPLAY_ROOT + "/")
                visual_display_proxy = path.startswith(VISUAL_DISPLAY_ROOT + "/")
                display_proxy = collision_display_proxy or visual_display_proxy
                if not display_proxy and root_path != "/" and path != root_path and not path.startswith(
                    root_path + "/"
                ):
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                collision = _usd_is_collision_prim(prim, UsdPhysics)
                collision_only = _usd_is_collision_only_prim(
                    prim, UsdGeom, UsdPhysics
                )
                if not display_proxy and (
                    path in visual_proxy_sources
                    or (path in proxy_sources and collision_only)
                ):
                    continue
                source_path = str(
                    prim.GetAttribute("blacknode:sourcePath").Get() or ""
                ) if display_proxy and prim.HasAttribute("blacknode:sourcePath") else ""
                entry = {
                    "path": path,
                    "source_path": source_path or path,
                    "visual": bool(
                        visual_display_proxy
                        or (
                            not collision_only
                            and not display_proxy
                            and path not in visual_proxy_sources
                        )
                    ),
                    "collider": bool(
                        collision_display_proxy
                        or (
                            collision
                            and not display_proxy
                            and path not in proxy_sources
                        )
                    ),
                }
                source_or_path = source_path or path
                body_index = next(
                    (
                        index
                        for body_path, index in body_paths
                        if source_or_path == body_path
                        or source_or_path.startswith(body_path + "/")
                    ),
                    -1,
                )
                if body_index >= 0:
                    shape_bind_world = render_xform_cache.GetLocalToWorldTransform(prim)
                    entry["initial_world_matrix"] = [
                        float(shape_bind_world[row][column])
                        for row in range(4)
                        for column in range(4)
                    ]
                    body_bind_world = body_bind_worlds[body_index]
                    entry["body_bind_world_matrix"] = [
                        float(body_bind_world[row][column])
                        for row in range(4)
                        for column in range(4)
                    ]
                    parent = prim.GetParent()
                    render_parent_world = (
                        render_xform_cache.GetLocalToWorldTransform(parent)
                        if parent
                        and parent.IsValid()
                        and parent.IsA(UsdGeom.Xformable)
                        else Gf.Matrix4d(1.0)
                    )
                    entry["render_parent_world_matrix"] = [
                        float(render_parent_world[row][column])
                        for row in range(4)
                        for column in range(4)
                    ]
                    entry["body_index"] = body_index
                render_shapes_by_path[path] = entry
        runtime_body_names = {
            str(body.get("name") or "")
            for body in list(self.scene.get("rigid_bodies") or [])
        }
        for shape_index, (raw_label, raw_flags, body_index) in enumerate(
            zip(self.model.shape_label, shape_flags, shape_bodies)
        ):
            label = str(raw_label or "")
            flags = int(raw_flags)
            if asset_format in {"urdf", "mjcf"}:
                render_path = f"/root/model/shapes/shape_{shape_index}/instance_0"
            elif label.startswith("/"):
                render_path = label
                if render_path in visual_proxy_sources or render_path in proxy_sources:
                    continue
            elif int(body_index) >= 0:
                body_label = str(self.model.body_label[int(body_index)] or "")
                if body_label not in runtime_body_names:
                    continue
                render_path = f"/BlacknodeOVRT/RigidBodies/body_{int(body_index)}"
            else:
                continue
            default_entry = {
                "path": render_path,
                "visual": False,
                "collider": False,
            }
            if asset_format in {"urdf", "mjcf"}:
                default_entry.update({
                    "shape_index": shape_index,
                    "body_index": int(body_index),
                })
            entry = render_shapes_by_path.setdefault(render_path, default_entry)
            collision_only = render_path in collision_only_paths
            entry["visual"] = bool(
                entry["visual"]
                or (
                    flags & visible_bit
                    and not collision_only
                    and render_path not in visual_proxy_sources
                )
            )
            entry["collider"] = bool(
                entry["collider"]
                or ((flags & collide_bit or collision_only) and render_path not in proxy_sources)
            )
        self.render_shapes = list(render_shapes_by_path.values())
        self._annotate_scene_item_physics()
        try:
            body_masses = self.model.body_mass.numpy().tolist()
            body_inertias = self.model.body_inertia.numpy().tolist()
        except Exception:
            body_masses = []
            body_inertias = []
        for name, dynamics in self.joint_dynamics.items():
            body_index = int(dynamics["child_body_index"])
            if 0 <= body_index < len(self.model.body_label):
                dynamics["child_body"] = str(
                    self.model.body_label[body_index] or f"body_{body_index}"
                )
            if 0 <= body_index < len(body_masses):
                dynamics["child_body_mass_kg"] = float(body_masses[body_index])
            if 0 <= body_index < len(body_inertias):
                inertia = body_inertias[body_index]
                dynamics["child_body_inertia_kg_m2"] = [
                    float(inertia[0][0]), float(inertia[1][1]), float(inertia[2][2])
                ]
        flags = self.model.shape_flags.numpy().tolist()
        collision_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        validated_collision_paths = [
            path
            for path in normalized_collision_paths
            if not any(pattern in path for pattern in disabled_shape_patterns)
        ]
        active_collision_indices, missing_collision_paths, inactive_collision_paths = (
            _normalized_collision_import_status(
                validated_collision_paths,
                dict(usd_import_result.get("path_shape_map") or {}),
                flags,
                collision_bit,
            )
        )
        # Include additional convex-decomposition parts in the status count.
        # Safety itself is checked through path_shape_map above; labels are only
        # used here to account for importer-generated parts in operator telemetry.
        source_labels = {
            str(self.model.shape_label[index])
            for index in active_collision_indices
        }
        self.active_collision_shapes = sum(
            1
            for shape_index, (label, flag) in enumerate(zip(self.model.shape_label, flags))
            if int(flag) & collision_bit
            and (
                shape_index in collision_proxy_shape_indices
                or str(label) in source_labels
                or any(str(label).startswith(source + "_convex_") for source in source_labels)
            )
        )
        if missing_collision_paths or inactive_collision_paths:
            details: list[str] = []
            if missing_collision_paths:
                details.append("not imported: " + ", ".join(missing_collision_paths))
            if inactive_collision_paths:
                details.append("collision disabled: " + ", ".join(inactive_collision_paths))
            raise RuntimeError(
                "Newton did not activate every normalized USD collision mesh; "
                "physics cannot start safely (" + "; ".join(details) + ")"
            )
        solver_name = str(self.scene.get("solver") or "xpbd").strip().lower()
        use_mujoco_native_contacts = bool(
            self.scene.get("mujoco_use_native_contacts", True)
        )
        self.solver_uses_native_contacts = (
            solver_name == "mujoco" and use_mujoco_native_contacts
        )
        requested_broad_phase = str(
            self.scene.get("collision_broad_phase") or "explicit"
        ).strip().lower()
        if requested_broad_phase not in {"explicit", "nxn", "sap"}:
            raise ValueError(
                "collision_broad_phase must be one of: explicit, nxn, sap"
            )
        if solver_name == "mujoco":
            use_mujoco_cpu = bool(self.scene.get("mujoco_cpu", False))
            solver_contact_budget = int(
                self.scene.get("rigid_contact_max_per_world")
                or self.model.rigid_contact_max
            )
            external_contact_limits = (
                {
                    "nconmax": solver_contact_budget,
                    # A 6-D frictional contact can contribute multiple scalar
                    # constraints. Keep enough EFC rows for every accepted
                    # Newton contact instead of MJWarp's small mesh default.
                    "njmax": max(256, solver_contact_budget * 6),
                }
                if not use_mujoco_native_contacts
                else {}
            )
            update_data_interval = int(
                self.scene.get(
                    "mujoco_update_data_interval", 0 if use_mujoco_cpu else 1
                )
            )
            if update_data_interval < 0:
                raise ValueError("mujoco_update_data_interval cannot be negative")
            if not use_mujoco_cpu:
                self.model.request_contact_attributes("force")
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=use_mujoco_native_contacts,
                use_mujoco_cpu=use_mujoco_cpu,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                iterations=self.solver_iterations,
                ls_iterations=100,
                ccd_iterations=max(1, int(self.scene.get("mujoco_ccd_iterations", 35))),
                enable_multiccd=bool(self.scene.get("mujoco_enable_multiccd", False)),
                impratio=float(self.scene.get("mujoco_impratio", 1.0)),
                update_data_interval=update_data_interval,
                save_to_mjcf=(
                    str(self.scene["mujoco_save_to_mjcf"])
                    if self.scene.get("mujoco_save_to_mjcf")
                    else None
                ),
                **external_contact_limits,
            )
        elif solver_name == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model, iterations=self.solver_iterations
            )
        else:
            raise ValueError(f"unsupported rigid solver: {solver_name!r}")
        self.collision_pipeline = None
        if requested_broad_phase == "explicit":
            self.contacts = self.model.contacts()
        else:
            contact_budget = int(
                self.scene.get("rigid_contact_max_per_world")
                or self.model.rigid_contact_max
            )
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase=requested_broad_phase,
                shape_pairs_max=contact_budget,
                rigid_contact_max=contact_budget,
            )
            self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()
        self._new_states(newton, wp)
        if asset_path and asset_format in {"urdf", "mjcf"}:
            self._export_model_render_asset(newton)
        provider = str(self.viewer_config.get("provider") or "viser")
        self.viewer = create_viewer(provider, self, self.model, self.viewer_config)
        self.viewer.set_render_options(self.show_visuals, self.show_colliders)

    def _annotate_scene_item_physics(self) -> None:
        """Connect Outliner prims to the movable Newton body that owns them."""
        body_paths = sorted(
            (
                (str(label).rstrip("/"), index)
                for index, label in enumerate(list(self.model.body_label))
                if str(label).startswith("/")
            ),
            key=lambda value: len(value[0]),
            reverse=True,
        )
        for item in self.scene_items:
            path = str(item.get("path") or "").rstrip("/")
            owner = next(
                (
                    (body_path, body_index)
                    for body_path, body_index in body_paths
                    if path == body_path or path.startswith(body_path + "/")
                ),
                None,
            )
            dynamic_descendants = [
                body_index
                for body_path, body_index in body_paths
                if body_index in self.dynamic_body_indices
                and (body_path == path or body_path.startswith(path + "/"))
            ]
            indices = [owner[1]] if owner is not None else dynamic_descendants
            if not indices:
                continue
            dynamic = bool(
                owner is None or int(owner[1]) in self.dynamic_body_indices
            )
            item["physics_dynamic"] = dynamic
            item["physics_body_indices"] = list(dict.fromkeys(indices))
            if owner is not None:
                item["physics_body_index"] = int(owner[1])
                item["physics_body_path"] = owner[0]
                # Only the actual free-body Xform teleports Newton. Descendant
                # meshes keep independent USD authoring semantics and ordinary
                # Xforms stay editable even when they group physics geometry.
                item["physics_pose_editable"] = bool(dynamic and path == owner[0])

    def _assign_body_pose_overrides(self, wp: Any) -> None:
        if not self.body_pose_overrides:
            return
        newton, _wp = self._imports()
        for state in (self.state_0, self.state_1):
            joint_q = getattr(state, "joint_q", None)
            joint_qd = getattr(state, "joint_qd", None)
            if joint_q is None or joint_qd is None:
                body_q = getattr(state, "body_q", None)
                body_qd = getattr(state, "body_qd", None)
                if body_q is None:
                    continue
                poses = body_q.numpy().tolist()
                velocities = body_qd.numpy().tolist() if body_qd is not None else []
                for index, pose in self.body_pose_overrides.items():
                    if not 0 <= index < len(poses):
                        continue
                    poses[index] = list(pose)
                    if 0 <= index < len(velocities):
                        velocities[index] = [0.0] * 6
                body_q.assign(
                    wp.array(poses, dtype=wp.transform, device=self.model.device)
                )
                if body_qd is not None:
                    body_qd.assign(
                        wp.array(
                            velocities,
                            dtype=wp.spatial_vector,
                            device=self.model.device,
                        )
                    )
                continue
            coordinates = joint_q.numpy().tolist()
            velocities = joint_qd.numpy().tolist()
            for index, pose in self.body_pose_overrides.items():
                starts = self.free_joint_state_starts.get(int(index))
                if starts is None:
                    continue
                q_start, qd_start = starts
                coordinates[q_start : q_start + 7] = list(pose)
                velocities[qd_start : qd_start + 6] = [0.0] * 6
            joint_q.assign(wp.array(coordinates, dtype=wp.float32, device=self.model.device))
            joint_qd.assign(wp.array(velocities, dtype=wp.float32, device=self.model.device))
            newton.eval_fk(self.model, joint_q, joint_qd, state)
        solver = getattr(self, "solver", None)
        if solver is not None:
            solver.reset(self.state_0, flags=0)
            # With continuous native MuJoCo ownership (update interval 0), a
            # teleport is an explicit synchronization point instead of being
            # copied again at every following physics step.
            if int(getattr(solver, "update_data_interval", 1)) != 1:
                solver_data = (
                    solver.mj_data
                    if bool(getattr(solver, "use_mujoco_cpu", False))
                    else solver.mjw_data
                )
                solver._update_mjc_data(solver_data, self.model, self.state_0)

    def _new_states(self, newton: Any, wp: Any) -> None:
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.reference_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._assign_body_pose_overrides(wp)
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.reference_state
        )
        try:
            body_positions = self.state_0.body_q.numpy().tolist()
            points = [
                [float(value) for value in body_positions[index][:3]]
                for index in sorted(self.articulation_body_indices)
                if 0 <= index < len(body_positions)
            ]
            extent = max(
                (
                    max(point[axis] for point in points)
                    - min(point[axis] for point in points)
                    for axis in range(3)
                ),
                default=0.0,
            )
            beside = max(0.35, extent * 1.25)
            self.digital_twin_ghost["beside_offset_m"] = [beside, 0.0, 0.0]
            if self.digital_twin_ghost.get("placement") == "beside":
                self.digital_twin_ghost["offset_m"] = [beside, 0.0, 0.0]
        except Exception:
            pass
        if self.external_joint_positions:
            self._update_reference_state()
        if self.control is None or self.control.joint_target_q is None:
            self.sim_time = 0.0
            self.frame_count = 0
            return
        targets = self.model.joint_target_q.numpy().tolist()
        for name, drive_index in self.joint_drive_indices.items():
            targets[drive_index] = self.applied[name]
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
        with self.lock:
            for name, index in self.joint_indices.items():
                motion = self.joint_motion_limits[name]
                max_delta = min(
                    float(motion["max_step"]), float(motion["max_velocity"]) * frame_dt
                )
                requested = self.desired[name]
                prior = self.applied[name]
                delta = min(max_delta, max(-max_delta, requested - prior))
                value = prior + delta
                lower, upper = self.joint_limits[name]
                value = min(upper, max(lower, value))
                self.applied[name] = value
                drive_index = self.joint_drive_indices.get(name, index)
                target_array[drive_index] = value
        self.control.joint_target_q.assign(
            wp.array(target_array, dtype=wp.float32, device=self.model.device)
        )

    def _read_state(self, newton: Any) -> None:
        if self.state_0.joint_q is None:
            return
        # SolverMuJoCo already converts mjData.qpos/qvel into Newton joint_q,
        # joint_qd and body_q in _update_newton_state(). Running eval_ik over
        # that result is redundant and can rewrite a free joint from its body
        # transform. In contact-heavy frames this corrupted the cube's X/Y
        # coordinates without a corresponding velocity, making it teleport in
        # the viewer on the first lift step. XPBD still needs body-to-joint IK.
        if not self.solver_uses_native_contacts:
            newton.eval_ik(
                self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd
            )
        coordinates = self.state_0.joint_q.numpy().tolist()
        with self.lock:
            self.current = {name: float(coordinates[index]) for name, index in self.joint_indices.items()}

    def _expire_stale_stream_follow(self, now: float | None = None) -> bool:
        """Disarm a stream-owned simulation command when its heartbeat expires."""
        checked_at = time.monotonic() if now is None else float(now)
        with self.lock:
            if not self.stream_follow_deadline or checked_at <= self.stream_follow_deadline:
                return False
            self.stream_follow_deadline = 0.0
            self.stream_follow_source = ""
            self.armed = False
            self.desired = dict(self.applied)
            return True

    def _loop(self) -> None:
        newton, wp = self._imports()
        frame_dt = 1.0 / self.fps
        step_dt = frame_dt / self.substeps
        next_frame = time.perf_counter()
        render_fps = max(
            1,
            min(self.fps, int(self.viewer_config.get("render_fps") or 30)),
        )
        render_interval = 1.0 / render_fps
        next_render = next_frame
        try:
            while not self.stop_event.is_set() and self.viewer.is_running():
                with self.lock:
                    self._expire_stale_stream_follow()
                    paused = self.paused
                    single_step = paused and self.pending_steps > 0
                    if single_step:
                        self.pending_steps -= 1
                    reset = self.reset_requested
                    self.reset_requested = False
                    if reset:
                        self._new_states(newton, wp)
                    if not paused or single_step:
                        # Serialize a physics step with viewport teleports. A
                        # drag can pause and replace body_q immediately after a
                        # complete frame, never halfway through solver writes.
                        self._apply_safe_target(frame_dt, wp)
                        for _substep_index in range(self.substeps):
                            self.state_0.clear_forces()
                            # Native-contact MuJoCo performs its own collision
                            # pass. In external-contact mode Newton generates
                            # triangle-mesh contacts and MJWarp consumes them.
                            if not self.solver_uses_native_contacts:
                                if self.collision_pipeline is None:
                                    self.model.collide(self.state_0, self.contacts)
                                else:
                                    self.collision_pipeline.collide(self.state_0, self.contacts)
                            self.solver.step(
                                self.state_0, self.state_1, self.control, self.contacts, step_dt
                            )
                            self.state_0, self.state_1 = self.state_1, self.state_0
                        self.sim_time += frame_dt
                        self.frame_count += 1
                        self._read_state(newton)
                render_now = time.perf_counter()
                if render_now >= next_render:
                    self.viewer.begin_frame(self.sim_time)
                    with self.lock:
                        self.viewer.log_state(self.state_0)
                        self.viewer.log_reference_state(
                            self.reference_state if self.external_joint_positions else None,
                            {
                                **copy.deepcopy(self.digital_twin_ghost),
                                "visible": bool(
                                    self.external_joint_positions
                                    and self.digital_twin_ghost.get("visible", True)
                                ),
                            },
                        )
                    self.viewer.end_frame()
                    next_render = max(
                        next_render + render_interval,
                        render_now + render_interval,
                    )
                next_frame += frame_dt
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_frame = time.perf_counter()
                    # A solver that is slightly slower than real time must
                    # still yield after each completed frame. Without this,
                    # the physics thread immediately reacquires `self.lock`
                    # and can starve native viewers, camera sensors, gizmos,
                    # and control requests for seconds at a time.
                    time.sleep(0)
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

    def record_joint_observation(
        self,
        positions: dict[str, Any],
        source: str,
        observed_at: Any = None,
        stale_after_seconds: Any = 0.5,
        *,
        compact: bool = False,
    ) -> dict[str, Any]:
        """Record read-only external joint telemetry for digital-twin comparison."""
        if not positions:
            raise ValueError("external joint observation must contain at least one joint")
        received_at = time.time()
        clean_observed_at = 0.0 if observed_at in {None, ""} else float(observed_at)
        if clean_observed_at and not math.isfinite(clean_observed_at):
            raise ValueError("external joint observation timestamp must be finite")
        clean_stale_after = float(stale_after_seconds)
        if not math.isfinite(clean_stale_after) or clean_stale_after <= 0.0:
            raise ValueError("external joint stale threshold must be positive and finite")
        clean: dict[str, float] = {}
        with self.lock:
            unknown = sorted(set(positions) - set(self.joint_indices))
            if unknown:
                raise ValueError("external observation contains unknown joint(s): " + ", ".join(unknown))
            for name, raw_value in positions.items():
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"external observation for joint {name!r} must be finite")
                clean[str(name)] = value
            self.external_joint_observation_source = str(source or "external")
            self.external_joint_observation_received_at = received_at
            self.external_joint_observation_observed_at = clean_observed_at
            self.external_joint_observation_stale_after = min(10.0, max(0.1, clean_stale_after))
            self.external_joint_positions = clean
            # Live Robot Monitor following hides the reference ghost. Avoid a
            # GPU FK allocation/evaluation for that invisible pose on every
            # USB sample; it can otherwise consume most of the stale-command
            # deadline and stall both physics and the renderer.
            if bool(self.digital_twin_ghost.get("visible", True)):
                self._update_reference_state()
            simulated = {name: float(self.current.get(name, 0.0)) for name in clean}
            errors = {name: simulated[name] - value for name, value in clean.items()}
            absolute_errors = [abs(value) for value in errors.values()]
            raw_latency = received_at - clean_observed_at if clean_observed_at else None
            source_latency = (
                max(0.0, raw_latency)
                if raw_latency is not None and abs(raw_latency) <= 24.0 * 60.0 * 60.0
                else None
            )
            sample = {
                "received_at": received_at,
                "observed_at": clean_observed_at or None,
                "source": self.external_joint_observation_source,
                "source_latency_seconds": source_latency,
                "joint_errors": errors,
                "max_abs_error": max(absolute_errors, default=0.0),
                "rms_error": math.sqrt(
                    sum(value * value for value in errors.values()) / max(1, len(errors))
                ),
            }
            if (
                self.digital_twin_history
                and received_at - float(self.digital_twin_history[-1]["received_at"])
                < DIGITAL_TWIN_HISTORY_INTERVAL_SECONDS
            ):
                self.digital_twin_history[-1] = sample
            else:
                self.digital_twin_history.append(sample)
            if len(self.digital_twin_history) > DIGITAL_TWIN_HISTORY_LIMIT:
                del self.digital_twin_history[:-DIGITAL_TWIN_HISTORY_LIMIT]
        return self.stream_control_status() if compact else self.status()

    def record_and_follow_joint_observation(
        self,
        positions: dict[str, Any],
        source: str,
        observed_at: Any = None,
        stale_after_seconds: Any = 0.5,
    ) -> dict[str, Any]:
        """Record and command one fresh stream sample as one ownership update."""
        # Keep the observation and command under one re-entrant lock so the
        # physics watchdog cannot expire stream ownership between recording a
        # fresh sample and renewing its deadline.
        with self.lock:
            self.record_joint_observation(
                positions,
                source=source,
                observed_at=observed_at,
                stale_after_seconds=stale_after_seconds,
                compact=True,
            )
            return self.follow_joint_observation(
                positions,
                source=source,
                stale_after_seconds=stale_after_seconds,
            )

    def _update_reference_state(self) -> None:
        if self.model is None or self.reference_state is None:
            return
        newton, wp = self._imports()
        coordinates = self.model.joint_q.numpy().tolist()
        for name, index in self.joint_indices.items():
            coordinates[index] = float(
                self.external_joint_positions.get(name, self.current.get(name, coordinates[index]))
            )
        velocities = wp.zeros(len(self.model.joint_qd), dtype=wp.float32, device=self.model.device)
        newton.eval_fk(
            self.model,
            wp.array(coordinates, dtype=wp.float32, device=self.model.device),
            velocities,
            self.reference_state,
        )

    def set_digital_twin_ghost(
        self, visible: Any, placement: Any, offset_m: Any = None, *, compact: bool = False
    ) -> dict[str, Any]:
        mode = str(placement or "beside").strip().lower()
        if mode not in {"overlay", "beside", "custom"}:
            raise ValueError("Digital Twin ghost placement must be overlay, beside, or custom")
        if mode == "overlay":
            offset = [0.0, 0.0, 0.0]
        elif mode == "beside":
            offset = list(self.digital_twin_ghost["beside_offset_m"])
        else:
            offset = _finite_vector("Digital Twin ghost offset_m", offset_m)
            if any(abs(value) > 100.0 for value in offset):
                raise ValueError("Digital Twin ghost offsets must stay within 100 metres")
        with self.lock:
            self.digital_twin_ghost["visible"] = bool(visible)
            self.digital_twin_ghost["placement"] = mode
            self.digital_twin_ghost["offset_m"] = offset
            if bool(visible) and self.external_joint_positions:
                self._update_reference_state()
        return self.stream_control_status() if compact else self.status()

    def sync_simulation_to_external_pose(self) -> dict[str, Any]:
        with self.lock:
            if not self.external_joint_positions:
                raise RuntimeError("no real/reference joint pose is available")
            age = max(0.0, time.time() - self.external_joint_observation_received_at)
            if age > self.external_joint_observation_stale_after:
                raise RuntimeError("real/reference joint pose is stale; synchronization was rejected")
            if not self.armed:
                raise RuntimeError("simulation motion is disarmed; arm before synchronizing")
            target = dict(self.external_joint_positions)
        return self.command(target, source="digital-twin-sync-once")

    def clear_digital_twin_history(self) -> dict[str, Any]:
        """Clear the diagnostic trace while retaining the latest read-only observation."""
        with self.lock:
            self.digital_twin_history.clear()
        return self.status()

    def create_digital_twin_artifact(
        self, name: Any = "", *, asset_path: str = "", scene_label: str = ""
    ) -> dict[str, Any]:
        """Snapshot the bounded observation trace as a portable Newton run artifact."""
        with self.lock:
            history = copy.deepcopy(self.digital_twin_history)
            if not history:
                raise ValueError("Digital Twin tracking history is empty")
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            artifact_id = f"newton-run-{uuid.uuid4().hex[:20]}"
            clean_name = str(name or "").strip()[:120] or f"Newton trace {created_at}"
            maximum = max(float(sample.get("max_abs_error") or 0.0) for sample in history)
            rms = math.sqrt(
                sum(float(sample.get("rms_error") or 0.0) ** 2 for sample in history)
                / len(history)
            )
            duration = max(
                0.0,
                float(history[-1].get("received_at") or 0.0)
                - float(history[0].get("received_at") or 0.0),
            )
            return {
                "kind": "blacknode.newton-run-artifact",
                "schema_version": 1,
                "artifact_id": artifact_id,
                "name": clean_name,
                "created_at": created_at,
                "path": "",
                "run_id": self.run_id,
                "source": self.external_joint_observation_source,
                "scene": {
                    "asset_path": str(asset_path or ""),
                    "label": str(scene_label or "")[:240],
                },
                "joint_names": list(self.joint_indices),
                "joint_units": dict(self.joint_units),
                "physics": {
                    "fps": self.fps,
                    "substeps": self.substeps,
                    "solver_iterations": self.solver_iterations,
                    "device": str(getattr(self.model, "device", self.device_request)),
                },
                "summary": {
                    "sample_count": len(history),
                    "duration_seconds": duration,
                    "max_abs_error": maximum,
                    "rms_error": rms,
                },
                "samples": history,
            }

    def load_digital_twin_baseline(self, artifact: dict[str, Any]) -> dict[str, Any]:
        samples = list(artifact.get("samples") or [])
        if not samples:
            raise ValueError("Newton run artifact has no tracking samples")
        artifact_units = dict(artifact.get("joint_units") or {})
        artifact_names = [str(name) for name in list(artifact.get("joint_names") or [])]
        observed_names = {
            str(name)
            for sample in samples if isinstance(sample, dict)
            for name in dict(sample.get("joint_errors") or {})
        }
        matching = [
            name for name in artifact_names
            if name in observed_names
            and name in self.joint_indices
            and artifact_units.get(name) == self.joint_units.get(name)
        ]
        if not matching:
            raise ValueError("Newton run artifact has no joints matching this articulation and its units")
        normalized_history: list[dict[str, Any]] = []
        for raw_sample in samples[-DIGITAL_TWIN_HISTORY_LIMIT:]:
            sample = dict(raw_sample or {})
            errors = {
                name: float(dict(sample.get("joint_errors") or {})[name])
                for name in matching if name in dict(sample.get("joint_errors") or {})
            }
            if not errors:
                continue
            absolute_errors = [abs(value) for value in errors.values()]
            normalized_history.append({
                **sample,
                "joint_errors": errors,
                "max_abs_error": max(absolute_errors),
                "rms_error": math.sqrt(
                    sum(value * value for value in errors.values()) / len(errors)
                ),
            })
        if not normalized_history:
            raise ValueError("Newton run artifact has no comparable tracking samples")
        normalized_summary = {
            **dict(artifact.get("summary") or {}),
            "sample_count": len(normalized_history),
            "max_abs_error": max(sample["max_abs_error"] for sample in normalized_history),
            "rms_error": math.sqrt(
                sum(sample["rms_error"] ** 2 for sample in normalized_history)
                / len(normalized_history)
            ),
        }
        with self.lock:
            self.digital_twin_baseline = {
                "artifact_id": str(artifact.get("artifact_id") or ""),
                "name": str(artifact.get("name") or "Newton trace")[:120],
                "created_at": str(artifact.get("created_at") or ""),
                "source": str(artifact.get("source") or ""),
                "matched_joint_names": matching,
                "summary": normalized_summary,
                "history": normalized_history,
            }
        return self.status()

    def clear_digital_twin_baseline(self) -> dict[str, Any]:
        with self.lock:
            self.digital_twin_baseline = {}
        return self.status()

    def _digital_twin_status(self, now: float) -> dict[str, Any]:
        if not self.external_joint_positions:
            return {
                "available": False,
                "source": "",
                "matched_joint_count": 0,
                "reference_positions": {},
                "simulated_positions": {},
                "joint_errors": {},
                "age_seconds": None,
                "source_latency_seconds": None,
                "stale_after_seconds": self.external_joint_observation_stale_after,
                "stale": True,
                "max_abs_error": 0.0,
                "rms_error": 0.0,
                "history": [],
                "history_limit": DIGITAL_TWIN_HISTORY_LIMIT,
                "baseline": copy.deepcopy(self.digital_twin_baseline),
                "ghost": copy.deepcopy(self.digital_twin_ghost),
            }
        reference = dict(self.external_joint_positions)
        simulated = {name: float(self.current.get(name, 0.0)) for name in reference}
        errors = {name: simulated[name] - value for name, value in reference.items()}
        absolute_errors = [abs(value) for value in errors.values()]
        age = max(0.0, now - self.external_joint_observation_received_at)
        observed_at = self.external_joint_observation_observed_at
        raw_latency = self.external_joint_observation_received_at - observed_at if observed_at else None
        source_latency = (
            max(0.0, raw_latency)
            if raw_latency is not None and abs(raw_latency) <= 24.0 * 60.0 * 60.0
            else None
        )
        return {
            "available": True,
            "source": self.external_joint_observation_source,
            "received_at": self.external_joint_observation_received_at,
            "observed_at": observed_at or None,
            "age_seconds": age,
            "source_latency_seconds": source_latency,
            "stale_after_seconds": self.external_joint_observation_stale_after,
            "stale": age > self.external_joint_observation_stale_after,
            "matched_joint_count": len(reference),
            "reference_positions": reference,
            "simulated_positions": simulated,
            "joint_errors": errors,
            "max_abs_error": max(absolute_errors, default=0.0),
            "rms_error": math.sqrt(
                sum(value * value for value in errors.values()) / max(1, len(errors))
            ),
            "history": copy.deepcopy(self.digital_twin_history),
            "history_limit": DIGITAL_TWIN_HISTORY_LIMIT,
            "baseline": copy.deepcopy(self.digital_twin_baseline),
            "ghost": copy.deepcopy(self.digital_twin_ghost),
        }

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
                self.stream_follow_deadline = 0.0
                self.stream_follow_source = ""
                self.stream_follow_authorized_source = ""
        return self.status()

    def follow_joint_observation(
        self,
        positions: dict[str, Any],
        source: str,
        stale_after_seconds: Any = 0.5,
    ) -> dict[str, Any]:
        """Follow fresh observations only while simulation motion is armed."""
        stale_after = float(stale_after_seconds)
        if not math.isfinite(stale_after) or stale_after <= 0.0:
            raise ValueError("stream follow stale threshold must be positive and finite")
        stale_after = min(10.0, max(0.1, stale_after))
        with self.lock:
            clean_source = str(source or "external-stream")
            owns_active_follow = bool(
                self.stream_follow_deadline
                and clean_source == self.stream_follow_source
            )
            authorized_to_recover = bool(
                clean_source
                and clean_source == self.stream_follow_authorized_source
            )
            if not self.armed or not owns_active_follow:
                if not authorized_to_recover:
                    if not self.armed:
                        self.stream_follow_deadline = 0.0
                        self.stream_follow_source = ""
                    status = self.stream_control_status()
                    status["accepted"] = False
                    return status
                # A stale gap disarms the simulation but does not revoke the
                # operator's still-enabled Drive authorization. A fresh sample
                # from that same source can safely reacquire simulation-only
                # ownership; explicit Stop/Disarm clears the authorization.
                self.armed = True
                self.desired = dict(self.current)
                self.applied = dict(self.current)
                self.stream_follow_source = clean_source
                self.stream_follow_stale_after = stale_after
                self.stream_follow_deadline = time.monotonic() + stale_after
            # Drive Newton is an explicit simulation-only authorization. Keep
            # the visible articulation advancing while that stream owns the
            # follower; the workspace Pause path revokes ownership below.
            self.paused = False
            self.phase = "running"
        result = self.command(positions, source=source, compact=True)
        with self.lock:
            self.stream_follow_source = str(source or "external-stream")
            self.stream_follow_stale_after = stale_after
            self.stream_follow_deadline = time.monotonic() + stale_after
        return result

    def start_stream_follow(
        self,
        source: str,
        stale_after_seconds: Any = 0.5,
    ) -> dict[str, Any]:
        """Explicitly arm and resume a simulation-only live joint follower."""
        clean_source = str(source or "").strip()
        if not clean_source:
            raise ValueError("stream follow source is required")
        stale_after = float(stale_after_seconds)
        if not math.isfinite(stale_after) or stale_after <= 0.0:
            raise ValueError("stream follow stale threshold must be positive and finite")
        stale_after = min(10.0, max(0.1, stale_after))
        with self.lock:
            if self.phase not in {"running", "paused"}:
                raise RuntimeError("simulation is not running")
            self.paused = False
            self.phase = "running"
            self.armed = True
            self.desired = dict(self.current)
            self.applied = dict(self.current)
            self.stream_follow_source = clean_source
            self.stream_follow_authorized_source = clean_source
            self.stream_follow_stale_after = stale_after
            # The first compact editor acknowledgement may still overlap the
            # next USB sample. Subsequent samples restore the strict deadline.
            self.stream_follow_deadline = time.monotonic() + max(2.0, stale_after)
        return self.stream_control_status()

    def stop_stream_follow(self, source: str = "") -> dict[str, Any]:
        """Release stream ownership and disarm only the matching live follower."""
        clean_source = str(source or "").strip()
        with self.lock:
            active = bool(self.stream_follow_deadline)
            authorized = bool(self.stream_follow_authorized_source)
            if clean_source and (
                (active and clean_source != self.stream_follow_source)
                or (
                    authorized
                    and clean_source != self.stream_follow_authorized_source
                )
            ):
                status = self.stream_control_status()
                status["accepted"] = False
                return status
            self.stream_follow_deadline = 0.0
            self.stream_follow_source = ""
            self.stream_follow_authorized_source = ""
            if active:
                self.armed = False
                self.desired = dict(self.applied)
        return self.stream_control_status()

    def command(
        self,
        positions: dict[str, Any],
        source: str = "blacknode",
        *,
        compact: bool = False,
    ) -> dict[str, Any]:
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
        status = self.stream_control_status() if compact else self.status()
        return {**status, "command_source": str(source), "clamped": clamped}

    def stream_control_status(self) -> dict[str, Any]:
        """Return the small acknowledgement used by latency-sensitive control streams."""
        with self.lock:
            return {
                "open": self.phase not in {"stopped", "fault"},
                "armed": bool(self.armed),
                "simulation_running": self.phase == "running" and not self.paused,
                "phase": str(self.phase),
                "accepted": not bool(self.last_error),
                "command_count": int(self.command_count),
                "clamped": list(self.clamped),
            }

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self.lock:
            self.paused = bool(paused)
            if not self.paused:
                self.pending_steps = 0
            self.phase = "paused" if self.paused else "running"
        return self.status()

    def request_step(self) -> dict[str, Any]:
        """Advance exactly one rendered physics frame while paused."""
        with self.lock:
            if not self.paused:
                raise RuntimeError("single-step requires a paused simulation")
            if self.phase not in {"running", "paused"}:
                raise RuntimeError("simulation is not running")
            self.pending_steps += 1
        return self.status()

    def request_reset(self) -> dict[str, Any]:
        with self.lock:
            self.armed = False
            self.stream_follow_deadline = 0.0
            self.stream_follow_source = ""
            self.stream_follow_authorized_source = ""
            self.desired = dict(self.home)
            self.applied = dict(self.home)
            self.reset_requested = True
        return self.status()

    def set_visibility(
        self,
        path: str,
        visible: bool,
        visibility_overrides: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Update renderer visibility without rebuilding physics or the viewer."""
        clean_path = str(path or "").strip()
        update = getattr(self.viewer, "set_visibility", None)
        if not callable(update) or update(clean_path, bool(visible)) is False:
            raise RuntimeError(
                f"viewer provider '{self.viewer_config.get('provider')}' does not support live object visibility"
            )
        overrides = dict(visibility_overrides or {clean_path: bool(visible)})
        with self.lock:
            for item in self.scene_items:
                item_path = str(item.get("path") or "")
                parts = [part for part in item_path.split("/") if part]
                ancestors = ["/" + "/".join(parts[:index]) for index in range(1, len(parts) + 1)]
                item["visible"] = all(overrides.get(ancestor, True) for ancestor in ancestors)
        return self.status()

    def _live_viewer_update(self, method: str, *args: Any) -> None:
        update = getattr(self.viewer, method, None)
        if not callable(update) or update(*args) is False:
            raise RuntimeError(
                f"viewer provider '{self.viewer_config.get('provider')}' does not support live {method.removeprefix('set_')} edits"
            )

    def set_grid(self, visible: bool) -> dict[str, Any]:
        self._live_viewer_update("set_grid", bool(visible))
        return self.status()

    def _scene_item_world_matrix_m(
        self, path: str, replacement: dict[str, Any] | None = None
    ) -> Any:
        from pxr import Gf

        items = {
            str(item.get("path") or ""): item
            for item in self.scene_items
            if str(item.get("path") or "")
        }
        visiting: set[str] = set()

        def world(item_path: str) -> Any:
            if not item_path or item_path == "/" or item_path not in items:
                return Gf.Matrix4d(1.0)
            if item_path in visiting:
                raise RuntimeError(f"scene transform hierarchy contains a cycle at {item_path}")
            visiting.add(item_path)
            item = items[item_path]
            transform = (
                replacement
                if item_path == path and replacement is not None
                else dict(item.get("transform") or {})
            )
            local = _editor_transform_matrix_m(transform, Gf)
            parent = world(str(item.get("parent_path") or "/"))
            visiting.remove(item_path)
            return local * parent

        return world(path)

    def set_transform(self, path: str, transform: dict[str, Any]) -> dict[str, Any]:
        clean = copy.deepcopy(transform)
        clean_path = str(path or "")
        physics_backed = False
        with self.lock:
            item = next(
                (
                    value
                    for value in self.scene_items
                    if str(value.get("path") or "") == clean_path
                ),
                None,
            )
            affected = [
                index
                for index, label in enumerate(list(self.model.body_label))
                if index in self.dynamic_body_indices
                and (
                    str(label).rstrip("/") == clean_path.rstrip("/")
                )
            ]
            if affected:
                physics_backed = True
                previous = dict((item or {}).get("transform") or {})
                old_scale = [float(value) for value in previous.get("scale", [1.0] * 3)]
                new_scale = [float(value) for value in clean.get("scale", [1.0] * 3)]
                if any(abs(before - after) > 1.0e-9 for before, after in zip(old_scale, new_scale)):
                    raise ValueError(
                        "A dynamic body's scale changes its collider and mass; edit collider size in "
                        "Physics properties instead. Move and Rotate remain live while paused."
                    )
                # A pose edit establishes a new spawn pose. Pausing here makes
                # viewport placement deterministic even if the object was
                # manipulated while a simulation was advancing.
                self.paused = True
                self.phase = "paused"
                from pxr import Gf

                old_world = self._scene_item_world_matrix_m(clean_path)
                new_world = self._scene_item_world_matrix_m(clean_path, clean)
                delta_world = old_world.GetInverse() * new_world
                poses = self.state_0.body_q.numpy().tolist()
                for index in affected:
                    if not 0 <= index < len(poses):
                        continue
                    body_world = _newton_pose_matrix_m(poses[index], Gf)
                    pose = _newton_pose_from_matrix_m(body_world * delta_world, Gf)
                    poses[index] = pose
                    self.body_pose_overrides[index] = pose
                _newton, wp = self._imports()
                del _newton
                self._assign_body_pose_overrides(wp)
            for item in self.scene_items:
                if item.get("path") == clean_path:
                    item["transform"] = clean
                    break
        viewer_transform = {**clean, "_physics_backed": physics_backed}
        self._live_viewer_update("set_transform", clean_path, viewer_transform)
        return self.status()

    def set_material(
        self, path: str, material_path: str, material: dict[str, Any]
    ) -> dict[str, Any]:
        clean = copy.deepcopy(material)
        self._live_viewer_update("set_material", str(path), str(material_path), clean)
        with self.lock:
            for item in self.scene_items:
                if item.get("path") == path:
                    item["material"] = clean
                    break
        return self.status()

    def set_environment(self, environment: dict[str, Any]) -> dict[str, Any]:
        self._live_viewer_update("set_environment", copy.deepcopy(environment))
        return self.status()

    def set_render_options(self, show_visuals: Any, show_colliders: Any) -> dict[str, Any]:
        clean_visuals = bool(show_visuals)
        clean_colliders = bool(show_colliders)
        self._live_viewer_update("set_render_options", clean_visuals, clean_colliders)
        with self.lock:
            self.show_visuals = clean_visuals
            self.show_colliders = clean_colliders
        return self.status()

    def set_joint_properties(self, name: str, stiffness: Any, damping: Any) -> dict[str, Any]:
        clean_stiffness = self._validate_drive_gain("stiffness", stiffness)
        clean_damping = self._validate_drive_gain("damping", damping)
        with self.lock:
            if name not in self.joint_indices:
                raise ValueError(f"articulation joint does not exist: {name or '<empty>'}")
            index = self.joint_drive_indices.get(name, self.joint_indices[name])
            newton, wp = self._imports()
            del newton
            ke = self.model.joint_target_ke.numpy().tolist()
            kd = self.model.joint_target_kd.numpy().tolist()
            ke[index] = clean_stiffness
            kd[index] = clean_damping
            self.model.joint_target_ke.assign(
                wp.array(ke, dtype=wp.float32, device=self.model.device)
            )
            self.model.joint_target_kd.assign(
                wp.array(kd, dtype=wp.float32, device=self.model.device)
            )
            self.joint_drive_gains[name] = {
                "stiffness": clean_stiffness,
                "damping": clean_damping,
            }
        return self.status()

    def set_joint_motion_limits(
        self, name: str, max_velocity: Any, max_step: Any
    ) -> dict[str, Any]:
        clean_velocity = float(max_velocity)
        clean_step = float(max_step)
        if not math.isfinite(clean_velocity) or clean_velocity <= 0.0:
            raise ValueError("max_velocity must be a positive finite number")
        if not math.isfinite(clean_step) or clean_step <= 0.0:
            raise ValueError("max_step must be a positive finite number")
        with self.lock:
            if name not in self.joint_indices:
                raise ValueError(f"articulation joint does not exist: {name or '<empty>'}")
            angular = self.joint_units[name] == "radians"
            velocity_ceiling = math.radians(720.0) if angular else 10.0
            step_ceiling = math.radians(90.0) if angular else 1.0
            if clean_velocity > velocity_ceiling:
                unit = "rad/s" if angular else "m/s"
                raise ValueError(f"max_velocity must not exceed {velocity_ceiling:g} {unit}")
            if clean_step > step_ceiling:
                unit = "rad" if angular else "m"
                raise ValueError(f"max_step must not exceed {step_ceiling:g} {unit}")
            self.joint_motion_limits[name] = {
                "max_velocity": clean_velocity,
                "max_step": clean_step,
            }
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
            now = time.time()
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
                "stream_follow": {
                    "active": bool(self.stream_follow_deadline),
                    "source": self.stream_follow_source,
                    "stale_after_seconds": self.stream_follow_stale_after,
                },
                "digital_twin": self._digital_twin_status(now),
                "joint_names": list(self.joint_indices),
                "joint_units": dict(self.joint_units),
                "positions": dict(self.current),
                "targets": dict(self.desired),
                "applied": dict(self.applied),
                "joint_limits": {name: list(bounds) for name, bounds in self.joint_limits.items()},
                "joint_drive_gains": {
                    name: dict(settings) for name, settings in self.joint_drive_gains.items()
                },
                "gravity_compensation": copy.deepcopy(self.gravity_compensation),
                "joint_dynamics": copy.deepcopy(self.joint_dynamics),
                "joint_motion_limits": copy.deepcopy(self.joint_motion_limits),
                "friction_override_matches": {
                    pattern: len(indices)
                    for pattern, indices in self.friction_override_matches.items()
                },
                "fps": self.fps,
                "substeps": self.substeps,
                "solver_iterations": self.solver_iterations,
                "scene_items": copy.deepcopy(self.scene_items),
                "render_shapes": copy.deepcopy(self.render_shapes),
                "show_visuals": self.show_visuals,
                "show_colliders": self.show_colliders,
                "rigid_body_positions_m": rigid_body_positions,
                "particle_count": int(getattr(self.model, "particle_count", 0) or 0),
                "particle_fill": copy.deepcopy(self.scene.get("particle_fill") or {}),
                "authored_dynamic_body_names": list(self.authored_dynamic_body_names),
                "authored_dynamic_body_count": len(self.authored_dynamic_body_names),
                "usd_mesh_count": self.usd_mesh_count,
                "usd_meshes_with_normals": self.usd_meshes_with_normals,
                "usd_collision_mesh_count": self.usd_collision_mesh_count,
                "active_collision_shapes": self.active_collision_shapes,
                "clamped": list(self.clamped),
                "last_error": self.last_error,
                "elapsed_seconds": max(0.0, now - self.started_at),
            }

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=3.0)
        if self.viewer is not None:
            self.viewer.close()
        if self._workspace_overlay_path:
            Path(self._workspace_overlay_path).unlink(missing_ok=True)
            self._workspace_overlay_path = ""
        if self._generated_render_path:
            Path(self._generated_render_path).unlink(missing_ok=True)
            self._generated_render_path = ""
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
                "substeps": max(1, min(MAX_PHYSICS_SUBSTEPS, int(substeps))),
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
            "particle_count": 0, "particle_fill": {},
            "digital_twin": {
                "available": False, "source": "", "matched_joint_count": 0,
                "reference_positions": {}, "simulated_positions": {}, "joint_errors": {},
                "age_seconds": None, "source_latency_seconds": None,
                "stale_after_seconds": 0.5, "stale": True,
                "max_abs_error": 0.0, "rms_error": 0.0,
                "history": [], "history_limit": DIGITAL_TWIN_HISTORY_LIMIT,
                "baseline": {},
                "ghost": {
                    "visible": True, "placement": "beside",
                    "offset_m": [0.35, 0.0, 0.0],
                    "beside_offset_m": [0.35, 0.0, 0.0],
                    "opacity": 0.28, "color_rgb": [0.18, 0.86, 1.0],
                },
            },
            "stream_follow": {
                "active": False, "source": "", "stale_after_seconds": 0.5,
            },
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
    if action == "step":
        return session.request_step()
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


def record_joint_observation(
    run_id: str,
    positions: dict[str, Any],
    source: str = "external",
    observed_at: Any = None,
    stale_after_seconds: Any = 0.5,
) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    return session.record_joint_observation(
        positions,
        source=source,
        observed_at=observed_at,
        stale_after_seconds=stale_after_seconds,
    )


def _mapped_external_joint_positions(
    session: NewtonSession,
    positions: dict[str, Any],
    units: Any,
    joint_map: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Map a normalized external pose into the loaded articulation's SI units."""
    source_units = str(units or "").strip().lower()
    unit_kind = (
        "angular_degrees"
        if source_units in {"degree", "degrees", "deg"}
        else "angular_radians"
        if source_units in {"radian", "radians", "rad"}
        else "linear_metres"
        if source_units in {"metre", "metres", "meter", "meters", "m"}
        else ""
    )
    if not unit_kind:
        raise ValueError(f"unsupported external joint position unit: {source_units or '<empty>'}")
    mapping = {
        str(source): str(destination)
        for source, destination in dict(joint_map or {}).items()
        if str(source) and str(destination)
    }
    explicit_map = bool(mapping)
    allowed = set(session.joint_indices)
    mapped: dict[str, float] = {}
    for raw_name, raw_value in dict(positions or {}).items():
        source = str(raw_name or "").strip()
        if not source or (explicit_map and source not in mapping):
            continue
        destination = mapping.get(source, source)
        if destination not in allowed:
            continue
        target_units = str(session.joint_units.get(destination) or "radians")
        if target_units == "radians" and unit_kind == "angular_degrees":
            value = math.radians(float(raw_value))
        elif target_units == "radians" and unit_kind == "angular_radians":
            value = float(raw_value)
        elif target_units == "metres" and unit_kind == "linear_metres":
            value = float(raw_value)
        else:
            raise ValueError(
                f"external {source_units} values are incompatible with Newton joint "
                f"'{destination}' ({target_units})"
            )
        if not math.isfinite(value):
            raise ValueError(f"external joint {source!r} must contain a finite position")
        if destination in mapped:
            raise ValueError(f"multiple external joints map to Newton joint '{destination}'")
        mapped[destination] = value
    if not mapped:
        raise ValueError("external pose has no joints matching the loaded Newton articulation")
    return mapped


def clear_joint_observation_history(run_id: str) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    return session.clear_digital_twin_history()


def save_digital_twin_artifact(
    run_id: str, name: Any = "", *, asset_path: str = "", scene_label: str = ""
) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    artifact = session.create_digital_twin_artifact(
        name, asset_path=asset_path, scene_label=scene_label
    )
    directory = _digital_twin_artifact_directory()
    artifact_path = directory / f"{artifact['artifact_id']}.json"
    artifact["path"] = str(artifact_path)
    summary = {
        "artifact_id": artifact["artifact_id"],
        "kind": artifact["kind"],
        "schema_version": artifact["schema_version"],
        "name": artifact["name"],
        "created_at": artifact["created_at"],
        "path": artifact["path"],
        "source": artifact["source"],
        "scene_label": str(dict(artifact.get("scene") or {}).get("label") or ""),
        "joint_names": list(artifact.get("joint_names") or []),
        "joint_units": dict(artifact.get("joint_units") or {}),
        **dict(artifact.get("summary") or {}),
    }
    with _DIGITAL_TWIN_ARTIFACT_LOCK:
        records = _artifact_index()
        _write_json_atomic(artifact_path, artifact)
        records[artifact["artifact_id"]] = summary
        _write_json_atomic(
            directory / "index.json",
            {"kind": "blacknode.newton-run-index", "schema_version": 1,
             "artifacts": list(records.values())},
        )
    return artifact


def load_digital_twin_baseline(run_id: str, artifact_id: Any) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    return session.load_digital_twin_baseline(_read_digital_twin_artifact(artifact_id))


def clear_digital_twin_baseline(run_id: str) -> dict[str, Any]:
    session = get_session(run_id)
    if session is None:
        raise RuntimeError(f"Newton session '{run_id}' is not running")
    return session.clear_digital_twin_baseline()


_WORKSPACE_LOCK = threading.RLock()
_WORKSPACE_SCENE_PATH = ""
_WORKSPACE_VIEWER_PROVIDER = "viser"
_WORKSPACE_SCENE_SPEC = make_empty_scene_spec()
_WORKSPACE_EDITOR_STATE = _default_workspace_editor_state()
_WORKSPACE_NOTICE = ""


def _workspace_viewer_config(provider: str = "viser") -> dict[str, Any]:
    environment = copy.deepcopy(_WORKSPACE_EDITOR_STATE["environment"])
    return {
        "kind": "blacknode.newton-viewer",
        "schema_version": 1,
        "provider": str(provider or "viser").strip().lower(),
        "host": "0.0.0.0",
        "port": 8080,
        "label": "Blacknode Newton",
        "background_color": str(environment["background_color"]),
        "show_grid": bool(_WORKSPACE_EDITOR_STATE["show_grid"]),
        "show_visuals": bool(_WORKSPACE_EDITOR_STATE["show_visuals"]),
        "show_colliders": bool(_WORKSPACE_EDITOR_STATE["show_colliders"]),
        "show_visuals": bool(_WORKSPACE_EDITOR_STATE["show_visuals"]),
        "show_colliders": bool(_WORKSPACE_EDITOR_STATE["show_colliders"]),
        "environment": environment,
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
    warning = _WORKSPACE_NOTICE
    if service_open and asset_path and mesh_count and collision_count == 0:
        warning = "This USD is visual-only; it has no authored mesh collision geometry."
    environment = copy.deepcopy(_WORKSPACE_EDITOR_STATE["environment"])
    environment["custom_hdri_supported"] = _WORKSPACE_VIEWER_PROVIDER == "ovrtx"
    if (
        service_open
        and environment.get("hdri_path")
        and _WORKSPACE_VIEWER_PROVIDER != "ovrtx"
    ):
        warning = "Custom HDRI files render with the OVRT viewer; Viser is using its selected preset."
    items = list(status.get("scene_items") or [])
    if service_open and not any(item.get("path") == WORKSPACE_GROUND_PATH for item in items):
        ground = dict(_WORKSPACE_SCENE_SPEC.get("ground") or {})
        ground_height = float(ground.get("height_m", 0.0))
        session = get_session(WORKSPACE_RUN_ID)
        viewer = getattr(session, "viewer", None)
        up_axis = str(getattr(viewer, "_up_axis", "z") or "z").lower()
        translation = [0.0, 0.0, 0.0]
        translation[{"x": 0, "y": 1, "z": 2}.get(up_axis, 2)] = ground_height
        default_transform = {
            "translate_m": translation,
            "rotate_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        default_material = {
            "base_color": [0.18, 0.2, 0.24],
            "metallic": 0.0,
            "roughness": 0.8,
            "opacity": 1.0,
        }
        ovrtx_editable = _WORKSPACE_VIEWER_PROVIDER == "ovrtx"
        items.append({
            "path": WORKSPACE_GROUND_PATH,
            "parent_path": "/",
            "name": "Ground",
            "type_name": "GroundPlane",
            "visible": bool(ground.get("enabled", True)),
            "editable": ovrtx_editable,
            "material_editable": ovrtx_editable,
            "transform": copy.deepcopy(
                _WORKSPACE_EDITOR_STATE["transforms"].get(
                    WORKSPACE_GROUND_PATH, default_transform
                )
            ),
            "material": copy.deepcopy(
                _WORKSPACE_EDITOR_STATE["materials"].get(
                    WORKSPACE_GROUND_PATH, default_material
                )
            ),
            "material_path": WORKSPACE_GROUND_MATERIAL_PATH if ovrtx_editable else "",
        })
    if service_open:
        light = copy.deepcopy(environment.get("distant_light") or {})
        key_supported = _WORKSPACE_VIEWER_PROVIDER == "ovrtx"
        key_enabled = bool(light.get("enabled", True))
        hdri_selected = bool(
            environment.get("hdri_path")
            or str(environment.get("hdri") or "none") != "none"
        )
        hdri_enabled = bool(environment.get("hdri_enabled", True))
        empty_transform = {
            "translate_m": [0.0, 0.0, 0.0],
            "rotate_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        empty_material = {
            "base_color": [1.0, 1.0, 1.0],
            "metallic": 0.0,
            "roughness": 0.0,
            "opacity": 1.0,
        }
        items.extend([
            {
                "path": WORKSPACE_LIGHTS_PATH,
                "parent_path": "/Blacknode",
                "name": "Lights",
                "type_name": "LightScope",
                "visible": (key_supported and key_enabled) or (hdri_selected and hdri_enabled),
                "visibility_editable": key_supported or hdri_selected,
                "editable": False,
                "material_editable": False,
                "transform": copy.deepcopy(empty_transform),
                "material": copy.deepcopy(empty_material),
                "light": {"kind": "scope"},
            },
            {
                "path": WORKSPACE_KEY_LIGHT_PATH,
                "parent_path": WORKSPACE_LIGHTS_PATH,
                "name": "Key / Sun",
                "type_name": "DistantLight",
                "visible": key_supported and key_enabled,
                "visibility_editable": key_supported,
                "available": key_supported,
                "editable": False,
                "material_editable": False,
                "transform": copy.deepcopy(empty_transform),
                "material": copy.deepcopy(empty_material),
                "light": {
                    "kind": "distant",
                    "enabled": key_enabled,
                    "intensity": float(light.get("intensity", 2500.0)),
                    "color": str(light.get("color") or "#fff2e0"),
                    "angle_deg": float(light.get("angle_deg", 4.0)),
                    "rotation_deg": [
                        float(value)
                        for value in list(light.get("rotation_deg") or [-35.0, 25.0, -25.0])
                    ],
                },
            },
            {
                "path": WORKSPACE_HDRI_LIGHT_PATH,
                "parent_path": WORKSPACE_LIGHTS_PATH,
                "name": "HDRI Environment",
                "type_name": "DomeLight",
                "visible": hdri_selected and hdri_enabled,
                "visibility_editable": hdri_selected,
                "available": hdri_selected,
                "editable": False,
                "material_editable": False,
                "transform": copy.deepcopy(empty_transform),
                "material": copy.deepcopy(empty_material),
                "light": {
                    "kind": "dome",
                    "enabled": hdri_enabled,
                    "selected": hdri_selected,
                    "intensity": float(environment.get("intensity", 1.0)),
                    "show_background": bool(environment.get("show_background", True)),
                    "hdri": str(environment.get("hdri") or "none"),
                    "hdri_path": str(environment.get("hdri_path") or ""),
                },
            },
        ])
    session = get_session(WORKSPACE_RUN_ID)
    viewer_selected = str(getattr(getattr(session, "viewer", None), "selected_path", "") or "")
    viewer_selected = _workspace_scene_path(viewer_selected)
    if viewer_selected and any(item.get("path") == viewer_selected for item in items):
        _WORKSPACE_EDITOR_STATE["selected_path"] = viewer_selected
    selected_path = str(_WORKSPACE_EDITOR_STATE.get("selected_path") or "")
    selected_item = next((item for item in items if item.get("path") == selected_path), None)
    digital_twin = dict(status.get("digital_twin") or {})
    reference_positions = dict(digital_twin.get("reference_positions") or {})
    joint_errors = dict(digital_twin.get("joint_errors") or {})
    joints = []
    for name in list(status.get("joint_names") or []):
        limits = list(dict(status.get("joint_limits") or {}).get(name) or [0.0, 0.0])
        gains = dict(dict(status.get("joint_drive_gains") or {}).get(name) or {})
        dynamics = dict(dict(status.get("joint_dynamics") or {}).get(name) or {})
        motion = dict(dict(status.get("joint_motion_limits") or {}).get(name) or {})
        joints.append({
            "name": name,
            "units": str(dict(status.get("joint_units") or {}).get(name) or "radians"),
            "position": float(dict(status.get("positions") or {}).get(name, 0.0)),
            "target": float(dict(status.get("targets") or {}).get(name, 0.0)),
            "applied_target": float(dict(status.get("applied") or {}).get(name, 0.0)),
            "reference_position": (
                float(reference_positions[name]) if name in reference_positions else None
            ),
            "tracking_error": float(joint_errors[name]) if name in joint_errors else None,
            "limits": [float(value) for value in limits],
            "stiffness": float(gains.get("stiffness", XPBD_DRIVE_STIFFNESS)),
            "damping": float(gains.get("damping", XPBD_DRIVE_DAMPING)),
            "child_body": str(dynamics.get("child_body") or ""),
            "child_body_mass_kg": float(dynamics.get("child_body_mass_kg", 0.0)),
            "child_body_inertia_kg_m2": [
                float(value)
                for value in list(dynamics.get("child_body_inertia_kg_m2") or [0.0, 0.0, 0.0])
            ],
            "passive_damping": float(dynamics.get("passive_damping", 0.0)),
            "max_velocity": float(motion.get("max_velocity", 0.0)),
            "max_step": float(motion.get("max_step", 0.0)),
        })
    try:
        digital_twin_artifacts = list_digital_twin_artifacts()
        digital_twin_artifact_error = ""
    except RuntimeError as exc:
        digital_twin_artifacts = []
        digital_twin_artifact_error = str(exc)
    return {
        **status,
        "kind": "blacknode.newton-workspace",
        "schema_version": 1,
        "open": service_open,
        "simulation_running": service_open and not bool(status.get("paused")),
        "asset_path": asset_path,
        "scene_label": Path(asset_path).name if asset_path else "Empty stage",
        "dynamic_body_count": dynamic_count,
        "show_grid": bool(_WORKSPACE_EDITOR_STATE["show_grid"]),
        "environment": environment,
        "scene_items": items,
        "selected_path": selected_path,
        "selected_item": selected_item,
        "joints": joints,
        "digital_twin_artifacts": digital_twin_artifacts,
        "digital_twin_artifact_error": digital_twin_artifact_error,
        "warning": warning,
    }


def _start_workspace(
    scene: dict[str, Any], asset_path: str = "", provider: str = "viser"
) -> dict[str, Any]:
    global _WORKSPACE_SCENE_PATH, _WORKSPACE_VIEWER_PROVIDER, _WORKSPACE_SCENE_SPEC
    prior = get_session(WORKSPACE_RUN_ID)
    if prior is not None:
        prior.stop()
    _WORKSPACE_SCENE_PATH = str(asset_path or "")
    _WORKSPACE_VIEWER_PROVIDER = str(provider or "viser").strip().lower()
    _WORKSPACE_SCENE_SPEC = copy.deepcopy(scene)
    session_scene = copy.deepcopy(scene)
    session_scene["workspace_edits"] = {
        "visibility": copy.deepcopy(_WORKSPACE_EDITOR_STATE["visibility"]),
        "transforms": copy.deepcopy(_WORKSPACE_EDITOR_STATE["transforms"]),
        "materials": copy.deepcopy(_WORKSPACE_EDITOR_STATE["materials"]),
    }
    try:
        start_session(
            WORKSPACE_RUN_ID,
            session_scene,
            _workspace_viewer_config(_WORKSPACE_VIEWER_PROVIDER),
            "auto",
            60,
            8,
            24,
            XPBD_DRIVE_STIFFNESS,
            XPBD_DRIVE_DAMPING,
            copy.deepcopy(_WORKSPACE_EDITOR_STATE["joint_drive_overrides"]),
            180.0,
            6.0,
        )
        session = get_session(WORKSPACE_RUN_ID)
        if session is not None:
            for name, limits in dict(
                _WORKSPACE_EDITOR_STATE.get("joint_motion_limits") or {}
            ).items():
                if name in session.joint_indices:
                    session.set_joint_motion_limits(
                        name, limits["max_velocity"], limits["max_step"]
                    )
        control_session(WORKSPACE_RUN_ID, "pause")
        return _workspace_status()
    except Exception:
        _WORKSPACE_SCENE_PATH = ""
        _WORKSPACE_VIEWER_PROVIDER = "viser"
        raise


def _require_workspace_item(path: Any, *, editable: bool = False) -> dict[str, Any]:
    clean_path = _workspace_scene_path(path)
    item = next(
        (value for value in list(_workspace_status().get("scene_items") or []) if value.get("path") == clean_path),
        None,
    )
    if item is None:
        raise ValueError(f"scene object does not exist: {clean_path or '<empty>'}")
    if editable and not bool(item.get("editable")):
        raise ValueError(f"scene object cannot be transformed: {clean_path}")
    return item


def control_workspace(action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Operate the node-independent Newton workspace used by the editor app."""
    global _WORKSPACE_SCENE_PATH, _WORKSPACE_VIEWER_PROVIDER
    global _WORKSPACE_SCENE_SPEC, _WORKSPACE_EDITOR_STATE, _WORKSPACE_NOTICE
    command = str(action or "status").strip().lower()
    values = dict(payload or {})
    with _WORKSPACE_LOCK:
        if command == "status":
            return _workspace_status()
        if command == "open":
            if _workspace_status()["open"]:
                return _workspace_status()
            _WORKSPACE_EDITOR_STATE = _default_workspace_editor_state()
            _WORKSPACE_NOTICE = ""
            scene = make_default_workspace_scene_spec()
            return _start_workspace(
                scene,
                str(scene["asset_path"]),
                provider=str(values.get("provider") or "viser"),
            )
        if command == "new":
            _WORKSPACE_EDITOR_STATE = _default_workspace_editor_state()
            _WORKSPACE_NOTICE = ""
            return _start_workspace(
                make_empty_scene_spec(), provider=str(values.get("provider") or _WORKSPACE_VIEWER_PROVIDER)
            )
        if command in {"open_asset", "open_usd"}:
            _WORKSPACE_EDITOR_STATE = _default_workspace_editor_state()
            _WORKSPACE_NOTICE = ""
            source = str(values.get("asset_path") or "").strip()
            suffix = Path(source).suffix.lower()
            if suffix in {".urdf", ".xacro"}:
                scene = make_robot_description_scene_spec(
                    asset_path=source,
                    fixed_base=bool(values.get("fixed_base", True)),
                    ground_enabled=bool(values.get("ground_enabled", True)),
                    ground_height=float(values.get("ground_height") or 0.0),
                    self_collisions=bool(values.get("self_collisions", False)),
                    show_colliders=bool(values.get("show_colliders", False)),
                    xacro_environment=values.get("xacro_environment"),
                    xacro_arguments=values.get("xacro_arguments"),
                )
            elif suffix in {".xml", ".mjcf"}:
                scene = make_mjcf_scene_spec(
                    asset_path=source,
                    fixed_base=(
                        bool(values["fixed_base"])
                        if "fixed_base" in values
                        else None
                    ),
                    ground_enabled=(
                        bool(values["ground_enabled"])
                        if "ground_enabled" in values
                        else None
                    ),
                    ground_height=float(values.get("ground_height") or 0.0),
                    self_collisions=bool(values.get("self_collisions", False)),
                    show_colliders=bool(values.get("show_colliders", False)),
                )
            else:
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
                scene = make_default_workspace_scene_spec()
                return _start_workspace(scene, str(scene["asset_path"]), provider=provider)
            return _start_workspace(copy.deepcopy(_WORKSPACE_SCENE_SPEC), _WORKSPACE_SCENE_PATH, provider=provider)
        if command == "select":
            path = _workspace_scene_path(values.get("path"))
            if path:
                _require_workspace_item(path)
            _WORKSPACE_EDITOR_STATE["selected_path"] = path
            session = get_session(WORKSPACE_RUN_ID)
            if session is not None and session.viewer is not None:
                # Keep status polling from restoring an older viewport pick
                # after the operator selects a different Outliner row.
                set_selection = getattr(session.viewer, "set_selection", None)
                if callable(set_selection):
                    set_selection(path)
                else:
                    setattr(session.viewer, "selected_path", path)
            return _workspace_status()
        if command == "set_grid":
            visible = bool(values.get("show_grid", True))
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_grid(visible)
            _WORKSPACE_EDITOR_STATE["show_grid"] = visible
            _WORKSPACE_NOTICE = "Grid visibility changed live."
            return _workspace_status()
        if command == "set_render_options":
            show_visuals = bool(values.get("show_visuals", True))
            show_colliders = bool(values.get("show_colliders", False))
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_render_options(show_visuals, show_colliders)
            _WORKSPACE_EDITOR_STATE["show_visuals"] = show_visuals
            _WORKSPACE_EDITOR_STATE["show_colliders"] = show_colliders
            _WORKSPACE_NOTICE = "Viewport geometry display changed live."
            return _workspace_status()
        if command == "set_visibility":
            item = _require_workspace_item(values.get("path"))
            visible = bool(values.get("visible"))
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            if str(item.get("path") or "") in {
                WORKSPACE_LIGHTS_PATH, WORKSPACE_KEY_LIGHT_PATH, WORKSPACE_HDRI_LIGHT_PATH
            }:
                if not bool(item.get("visibility_editable", True)):
                    raise ValueError(f"light visibility is unavailable: {item['path']}")
                environment = copy.deepcopy(_WORKSPACE_EDITOR_STATE["environment"])
                distant = copy.deepcopy(environment.get("distant_light") or {})
                if item["path"] in {WORKSPACE_LIGHTS_PATH, WORKSPACE_KEY_LIGHT_PATH}:
                    distant["enabled"] = visible
                    environment["distant_light"] = distant
                if item["path"] in {WORKSPACE_LIGHTS_PATH, WORKSPACE_HDRI_LIGHT_PATH}:
                    environment["hdri_enabled"] = visible
                session.set_environment(environment)
                _WORKSPACE_EDITOR_STATE["environment"] = environment
                _WORKSPACE_NOTICE = "Light visibility changed live."
                return _workspace_status()
            proposed_visibility = copy.deepcopy(_WORKSPACE_EDITOR_STATE["visibility"])
            if item.get("type_name") != "GroundPlane":
                proposed_visibility[str(item["path"])] = visible
            session.set_visibility(
                str(item["path"]), visible, proposed_visibility
            )
            if item.get("type_name") == "GroundPlane":
                _WORKSPACE_SCENE_SPEC.setdefault("ground", {})["enabled"] = visible
            else:
                _WORKSPACE_EDITOR_STATE["visibility"] = proposed_visibility
            _WORKSPACE_NOTICE = "Visibility changed live; the viewer and simulation stayed active."
            return _workspace_status()
        if command == "set_transform":
            item = _require_workspace_item(values.get("path"), editable=True)
            transform = {
                "translate_m": _finite_vector("translate_m", values.get("translate_m")),
                "rotate_deg": _finite_vector("rotate_deg", values.get("rotate_deg")),
                "scale": _finite_vector("scale", values.get("scale"), positive=True),
            }
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_transform(str(item["path"]), transform)
            _WORKSPACE_EDITOR_STATE["transforms"][str(item["path"])] = transform
            _WORKSPACE_NOTICE = (
                "Dynamic body placed and physics paused; Play starts from this pose."
                if bool(item.get("physics_pose_editable"))
                else "Transform changed live."
            )
            return _workspace_status()
        if command == "set_material":
            item = _require_workspace_item(values.get("path"))
            if not bool(item.get("material_editable")):
                raise ValueError(f"scene object does not expose an editable material: {item['path']}")
            color = _finite_vector("base_color", values.get("base_color"))
            if not all(0.0 <= value <= 1.0 for value in color):
                raise ValueError("base_color values must be between zero and one")
            material = {"base_color": color}
            for name, default in (("metallic", 0.0), ("roughness", 0.5), ("opacity", 1.0)):
                value = float(values.get(name, default))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be between zero and one")
                material[name] = value
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_material(
                str(item["path"]),
                str(item.get("material_path") or _workspace_material_path(str(item["path"]))),
                material,
            )
            _WORKSPACE_EDITOR_STATE["materials"][str(item["path"])] = material
            _WORKSPACE_NOTICE = "Material changed live."
            return _workspace_status()
        if command == "set_environment":
            environment = _updated_workspace_environment(
                _WORKSPACE_EDITOR_STATE["environment"], values
            )
            _WORKSPACE_EDITOR_STATE["environment"] = environment
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_environment(environment)
            _WORKSPACE_NOTICE = "Environment changed live."
            return _workspace_status()
        if command == "set_light":
            item = _require_workspace_item(values.get("path"))
            if item.get("type_name") != "DistantLight":
                raise ValueError(f"scene item is not an editable distant light: {item['path']}")
            environment = _updated_workspace_environment(
                _WORKSPACE_EDITOR_STATE["environment"],
                {"distant_light": values},
            )
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_environment(environment)
            _WORKSPACE_EDITOR_STATE["environment"] = environment
            _WORKSPACE_NOTICE = "Distant light changed live."
            return _workspace_status()
        if command in {"arm", "disarm"}:
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_armed(command == "arm")
            return _workspace_status()
        if command == "start_robot_monitor_follow":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            source = str(values.get("source") or "").strip()
            stale_after = float(values.get("stale_after_seconds") or 0.5)
            session.set_digital_twin_ghost(False, "overlay", compact=True)
            session.start_stream_follow(source, stale_after)
            return session.stream_control_status()
        if command == "robot_monitor_home":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            if values.get("calibrated") is not True:
                raise RuntimeError("calibration home requires an active calibration")
            positions = _mapped_external_joint_positions(
                session,
                dict(values.get("positions") or {}),
                values.get("position_unit") or "degree",
                dict(values.get("joint_map") or {}),
            )
            source = str(values.get("source") or "robot-monitor:calibration-home").strip()
            session.stop_stream_follow("")
            session.set_digital_twin_ghost(False, "overlay")
            session.set_paused(False)
            session.set_armed(True)
            session.command(positions, source=source or "robot-monitor:calibration-home")
            return _workspace_status()
        if command == "robot_monitor_sample":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            if not bool(values.get("available", True)):
                raise RuntimeError("robot monitor telemetry is unavailable")
            if bool(values.get("stale")):
                raise RuntimeError("robot monitor telemetry is stale")
            if not bool(values.get("connected", True)):
                raise RuntimeError("robot monitor hardware is disconnected")
            if values.get("calibrated") is not True:
                raise RuntimeError("robot monitor telemetry must use an active calibration")
            stale_after = float(values.get("stale_after_seconds") or 0.5)
            if not math.isfinite(stale_after) or stale_after <= 0.0:
                raise ValueError("robot monitor stale threshold must be positive and finite")
            source_age = float(values.get("age_seconds") or 0.0)
            if not math.isfinite(source_age) or source_age < 0.0:
                raise ValueError("robot monitor telemetry age must be finite and non-negative")
            if source_age > stale_after:
                raise RuntimeError("robot monitor telemetry is older than its stale threshold")
            positions = _mapped_external_joint_positions(
                session,
                dict(values.get("positions") or {}),
                values.get("position_unit"),
                dict(values.get("joint_map") or {}),
            )
            calibration_home = {}
            if values.get("home_positions"):
                calibration_home = _mapped_external_joint_positions(
                    session,
                    dict(values.get("home_positions") or {}),
                    values.get("home_position_unit") or "degree",
                    dict(values.get("joint_map") or {}),
                )
            # Calibrated Robot Monitor positions are displacements from each
            # physical joint's saved home_ticks. Apply each displacement to
            # that calibration's measured home_offset_deg when supplied, with
            # the USD-authored home as a compatibility fallback. Recalculate
            # from the fixed baseline on every frame; never accumulate.
            positions = {
                name: float(calibration_home.get(name, session.home.get(name, 0.0))) + value
                for name, value in positions.items()
            }
            source = str(values.get("source") or "robot-monitor").strip() or "robot-monitor"
            observed_at = values.get("observed_at")
            if bool(values.get("follow", True)):
                session.set_digital_twin_ghost(False, "overlay", compact=True)
                return session.record_and_follow_joint_observation(
                    positions,
                    source=source,
                    observed_at=observed_at,
                    stale_after_seconds=stale_after,
                )
            session.record_joint_observation(
                positions,
                source=source,
                observed_at=observed_at,
                stale_after_seconds=stale_after,
                compact=True,
            )
            return session.stream_control_status()
        if command == "stop_robot_monitor_follow":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                return _workspace_status()
            session.stop_stream_follow(str(values.get("source") or ""))
            return session.stream_control_status()
        if command == "joint_target":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            name = str(values.get("name") or "").strip()
            session.command({name: values.get("target")}, source="editor robot controller")
            return _workspace_status()
        if command == "set_digital_twin_ghost":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.set_digital_twin_ghost(
                values.get("visible", True),
                values.get("placement", "beside"),
                values.get("offset_m"),
            )
            _WORKSPACE_NOTICE = "Real-pose ghost display changed live."
            return _workspace_status()
        if command == "sync_digital_twin_pose":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.sync_simulation_to_external_pose()
            _WORKSPACE_NOTICE = "Newton synchronized once to the fresh real/reference pose."
            return _workspace_status()
        if command == "clear_digital_twin_history":
            session = get_session(WORKSPACE_RUN_ID)
            if session is None:
                raise RuntimeError("Newton workspace is not open")
            session.clear_digital_twin_history()
            _WORKSPACE_NOTICE = "Digital Twin tracking history cleared."
            return _workspace_status()
        if command == "save_digital_twin_artifact":
            scene_label = Path(_WORKSPACE_SCENE_PATH).name if _WORKSPACE_SCENE_PATH else "Empty stage"
            artifact = save_digital_twin_artifact(
                WORKSPACE_RUN_ID,
                values.get("name"),
                asset_path=_WORKSPACE_SCENE_PATH,
                scene_label=scene_label,
            )
            _WORKSPACE_NOTICE = f"Saved Digital Twin run artifact: {artifact['name']}"
            return {**_workspace_status(), "saved_artifact": artifact}
        if command == "load_digital_twin_baseline":
            load_digital_twin_baseline(WORKSPACE_RUN_ID, values.get("artifact_id"))
            _WORKSPACE_NOTICE = "Loaded a read-only Digital Twin comparison baseline."
            return _workspace_status()
        if command == "clear_digital_twin_baseline":
            clear_digital_twin_baseline(WORKSPACE_RUN_ID)
            _WORKSPACE_NOTICE = "Digital Twin comparison baseline cleared."
            return _workspace_status()
        if command == "set_joint_properties":
            session = get_session(WORKSPACE_RUN_ID)
            name = str(values.get("name") or "").strip()
            if session is None or name not in session.joint_indices:
                raise ValueError(f"articulation joint does not exist: {name or '<empty>'}")
            settings = {
                "stiffness": NewtonSession._validate_drive_gain("stiffness", values.get("stiffness")),
                "damping": NewtonSession._validate_drive_gain("damping", values.get("damping")),
            }
            session.set_joint_properties(name, settings["stiffness"], settings["damping"])
            _WORKSPACE_EDITOR_STATE["joint_drive_overrides"][name] = settings
            _WORKSPACE_NOTICE = "Joint drive properties changed live."
            return _workspace_status()
        if command == "set_joint_motion":
            session = get_session(WORKSPACE_RUN_ID)
            name = str(values.get("name") or "").strip()
            if session is None or name not in session.joint_indices:
                raise ValueError(f"articulation joint does not exist: {name or '<empty>'}")
            session.set_joint_motion_limits(
                name, values.get("max_velocity"), values.get("max_step")
            )
            _WORKSPACE_EDITOR_STATE["joint_motion_limits"][name] = copy.deepcopy(
                session.joint_motion_limits[name]
            )
            _WORKSPACE_NOTICE = "Joint command rate changed live."
            return _workspace_status()
        if command in {"play", "start"}:
            if not _workspace_status()["open"]:
                scene = make_default_workspace_scene_spec()
                _start_workspace(scene, str(scene["asset_path"]))
            control_session(WORKSPACE_RUN_ID, "resume")
            return _workspace_status()
        if command in {"stop", "pause"}:
            if _workspace_status()["open"]:
                session = get_session(WORKSPACE_RUN_ID)
                if session is not None:
                    session.stop_stream_follow("")
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
            _WORKSPACE_SCENE_SPEC = make_empty_scene_spec()
            _WORKSPACE_EDITOR_STATE = _default_workspace_editor_state()
            _WORKSPACE_NOTICE = ""
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
    global _WORKSPACE_SCENE_PATH, _WORKSPACE_VIEWER_PROVIDER
    global _WORKSPACE_SCENE_SPEC, _WORKSPACE_EDITOR_STATE, _WORKSPACE_NOTICE
    with _LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
        hooks = list(_SHUTDOWN_HOOKS.items())
    _WORKSPACE_SCENE_PATH = ""
    _WORKSPACE_VIEWER_PROVIDER = "viser"
    _WORKSPACE_SCENE_SPEC = make_empty_scene_spec()
    _WORKSPACE_EDITOR_STATE = _default_workspace_editor_state()
    _WORKSPACE_NOTICE = ""
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
