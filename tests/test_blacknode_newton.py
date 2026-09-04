from __future__ import annotations

import json
import importlib.util
import io
import math
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BLACKNODE_ROOT = PACKAGE_ROOT.parents[1]
CORE_PYTHON = BLACKNODE_ROOT / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from blacknode.node import _NODE_REGISTRY  # noqa: E402
from blacknode.packages import load_package  # noqa: E402
from blacknode.workflow import graph_from_workflow, validate_workflow  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(predicate, timeout: float, message: str):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"{message}; last value: {last!r}")


class BlacknodeNewtonContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(
            PACKAGE_ROOT,
            component_overrides={
                "viewer-ovrtx": False, "rosbridge": False, "replay": False,
            },
        )

    def test_default_package_loads_real_runtime_and_browser_provider(self) -> None:
        self.assertTrue(self.package.ok, self.package.error)
        manifest = (PACKAGE_ROOT / "blacknode-package.toml").read_text(encoding="utf-8")
        self.assertIn('"opencv-python>=4.10,<6"', manifest)
        self.assertIn('"newton[importers,sim]>=1.4,<2"', manifest)
        self.assertIn('"mujoco_warp"', manifest)
        self.assertNotIn("opencv-python-headless", manifest)
        self.assertEqual(self.package.enabled_components, ["runtime", "viewer-viser"])
        self.assertEqual(
            set(self.package.node_types),
            {
                "NewtonJointCommand",
                "NewtonScene",
                "NewtonSimulation",
                "NewtonUSDScene",
                "NewtonViewerConfig",
                "SO101ReachTask",
            },
        )
        from blacknode.pkg.blacknode_newton.viewer_contract import available_viewers

        self.assertEqual(available_viewers(), ["viser"])
        self.assertIn("viewer-ovrtx", self.package.components)
        self.assertFalse(self.package.components["viewer-ovrtx"]["enabled"])

    def test_so101_reach_contract_is_simulation_only_and_profile_aligned(self) -> None:
        from blacknode.pkg.blacknode_newton.rl import ACTION_DIM, OBSERVATION_DIM, environment_spec

        spec = environment_spec(environment_count=32, episode_steps=64)
        self.assertEqual(spec["kind"], "blacknode.rl-environment")
        self.assertEqual(spec["robot_profile"], "so_arm101")
        self.assertEqual(spec["observation"]["dimension"], OBSERVATION_DIM)
        self.assertEqual(spec["action"]["dimension"], ACTION_DIM)
        self.assertTrue(spec["safety"]["simulation_only"])
        self.assertFalse(spec["safety"]["physical_motion_authorized"])
        result = _NODE_REGISTRY["SO101ReachTask"]({"environment_count": 32, "episode_steps": 64})
        self.assertIn("32 simulated arms", result["report"])
        self.assertEqual(result["environment"]["joint_names"], list(spec["joint_names"]))

    def test_training_batch_release_preserves_lightweight_preview(self) -> None:
        from blacknode.pkg.blacknode_newton.rl import SO101ReachEnvironment

        class Preview:
            url = "http://127.0.0.1:8091"

            def __init__(self) -> None:
                self.closed = False

            def is_running(self) -> bool:
                return not self.closed

            def close(self) -> None:
                self.closed = True

        environment = object.__new__(SO101ReachEnvironment)
        environment.environment_count = 512
        environment.preview_environment_index = 7
        environment.preview_frame = 42
        environment.preview_error = ""
        environment.preview_viewer = Preview()
        environment.preview_model = object()
        environment.preview_state = object()
        for name in (
            "template", "control", "contacts", "goal_state", "state_0", "state_1",
            "solver", "view", "model", "lower", "upper", "mid", "half_range",
            "previous_actions", "targets", "step_counts", "previous_distance",
        ):
            setattr(environment, name, object())

        status = environment.release_training_batch()

        self.assertTrue(status["running"])
        self.assertEqual(status["viewer_url"], "http://127.0.0.1:8091")
        self.assertIsNone(environment.model)
        self.assertIsNotNone(environment.preview_model)
        environment.close_preview()
        self.assertTrue(environment.preview_viewer is None)
        self.assertFalse(status.get("physical_motion_authorized", True))

    def test_training_preview_session_exposes_lightweight_viewer_scene(self) -> None:
        from blacknode.pkg.blacknode_newton.rl import _TrainingPreviewSession

        environment = SimpleNamespace(
            preview_frame=12,
            preview_model=SimpleNamespace(body_count=8),
            environment_count=512,
        )
        state = object()
        session = _TrainingPreviewSession(
            environment, state, PACKAGE_ROOT / "assets" / "so101_robot.usd"
        )

        self.assertIs(session.state_0, state)
        self.assertEqual(session.frame_count, 12)
        self.assertEqual(session.environment_count, 512)
        self.assertEqual(session.articulation_body_indices, set(range(8)))
        self.assertEqual(session.scene["asset_format"], "usd")
        self.assertTrue(Path(session.render_asset_path).is_file())

    def test_ovrtx_builds_live_bindings_for_lightweight_usd_sessions(self) -> None:
        provider_path = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
        )
        spec = importlib.util.spec_from_file_location(
            "blacknode_newton_ovrtx_training_bindings_test", provider_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from pxr import Usd, UsdGeom

        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "training-preview.usda"
            stage = Usd.Stage.CreateNew(str(asset))
            UsdGeom.Xform.Define(stage, "/Robot")
            UsdGeom.Xform.Define(stage, "/Robot/Link")
            UsdGeom.Cube.Define(stage, "/Robot/Link/visuals/body")
            UsdGeom.Cube.Define(stage, "/Robot/Link/collisions/body")
            stage.GetRootLayer().Save()

            entries = module._usd_bound_render_shapes(
                SimpleNamespace(body_label=["/Robot/Link"]), str(asset)
            )

        self.assertEqual(len(entries), 2)
        self.assertEqual({entry["body_index"] for entry in entries}, {0})
        self.assertEqual(sum(bool(entry["visual"]) for entry in entries), 1)
        self.assertEqual(sum(bool(entry["collider"]) for entry in entries), 1)
        self.assertTrue(all(len(entry["body_bind_world_matrix"]) == 16 for entry in entries))

    def test_so101_auto_device_requires_both_warp_and_torch_cuda(self) -> None:
        from blacknode.pkg.blacknode_newton.rl import _resolve_compute_device

        class FakeWarp:
            @staticmethod
            def is_cuda_available() -> bool:
                return True

            @staticmethod
            def get_device(value: str) -> str:
                return value

        cpu_torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        cuda_torch = SimpleNamespace(
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=lambda: True),
        )

        self.assertEqual(_resolve_compute_device(cpu_torch, FakeWarp, "auto"), "cpu")
        self.assertEqual(_resolve_compute_device(cuda_torch, FakeWarp, "auto"), "cuda:0")
        with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
            _resolve_compute_device(cpu_torch, FakeWarp, "cuda")

    def test_viser_training_preview_is_read_only_and_selects_sampled_environment(self) -> None:
        provider = (
            PACKAGE_ROOT / "components" / "viewer-viser" / "nodes" / "provider.py"
        ).read_text(encoding="utf-8")
        preview = provider.split("class ViserTrainingViewer:", 1)[1].split("def _factory", 1)[0]
        self.assertIn('"Environment index"', preview)
        self.assertIn('"Show target"', preview)
        self.assertIn('"Show end-effector trail"', preview)
        self.assertIn("physical motion disarmed", preview)
        self.assertNotIn("Arm joint commands", preview)
        self.assertIn('config.get("mode")', provider)

    def test_normalized_usd_colliders_use_newton_import_map_not_shape_labels(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        active, missing, inactive = runtime._normalized_collision_import_status(
            {
                "/World/instance/collisions/mesh",
                "/__BlacknodeColliderDisplay/Geometry/collider_085bb4f8109af169",
            },
            {"/World/instance/collisions/mesh": 1},
            [0, 4],
            4,
        )

        self.assertEqual(active, {1})
        self.assertEqual(missing, [])
        self.assertEqual(inactive, [])
        self.assertTrue(runtime._is_collider_display_path("/__BlacknodeColliderDisplay"))
        self.assertTrue(runtime._is_collider_display_path(
            "/__BlacknodeColliderDisplay/Geometry/collider_085bb4f8109af169"
        ))
        self.assertFalse(runtime._is_collider_display_path("/World/Collider"))

    def test_generated_convex_collision_hulls_keep_physics_but_lose_visibility(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        labels = [
            "/World/container",
            "/World/container_convex_1",
            "/World/container_convex_2",
            "/World/other_convex_1",
        ]
        flags = [7, 7, 7, 7]

        hidden = runtime._hide_generated_convex_collision_visuals(
            labels, flags, {"/World/container"}, 1
        )

        self.assertEqual(hidden, [1, 2])
        self.assertEqual(flags, [7, 6, 6, 7])
        self.assertTrue(flags[1] & 2)
        self.assertTrue(flags[1] & 4)

    def test_articulation_ghost_body_set_stays_on_connected_robot_component(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        bodies = runtime._connected_body_indices(
            {2},
            [-1, 0, 1, -1, 3],
            [0, 1, 2, 3, 4],
        )

        self.assertEqual(bodies, {0, 1, 2})

    def test_normalized_usd_collider_status_identifies_missing_and_inactive_paths(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        active, missing, inactive = runtime._normalized_collision_import_status(
            {"/World/collisions/active", "/World/collisions/inactive", "/World/collisions/missing"},
            {"/World/collisions/active": 0, "/World/collisions/inactive": 1},
            [4, 1],
            4,
        )

        self.assertEqual(active, {0})
        self.assertEqual(missing, ["/World/collisions/missing"])
        self.assertEqual(inactive, ["/World/collisions/inactive"])

    def test_optional_ovrtx_component_registers_separately_when_available(self) -> None:
        if importlib.util.find_spec("ovrtx") is None or importlib.util.find_spec("ovstage") is None:
            self.skipTest("optional OVRT release train is not installed")
        package = load_package(
            PACKAGE_ROOT,
            component_overrides={"viewer-ovrtx": True, "rosbridge": False},
        )
        self.assertTrue(package.ok, package.error)
        self.assertIn("viewer-ovrtx", package.enabled_components)
        from blacknode.pkg.blacknode_newton.viewer_contract import available_viewers

        self.assertIn("ovrtx", available_viewers())

    def test_rosbridge_maps_only_corresponding_newton_joints(self) -> None:
        bridge_path = PACKAGE_ROOT / "components" / "rosbridge" / "nodes" / "bridge.py"
        spec = importlib.util.spec_from_file_location("blacknode_newton_rosbridge_test", bridge_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        mapped = module._mapped_joint_positions(
            {"name": ["real_shoulder", "camera_tilt"], "position": [0.4, 1.2]},
            {"real_shoulder": "shoulder_lift"},
            {"shoulder_lift", "elbow_flex"},
        )
        self.assertEqual(mapped, {"shoulder_lift": 0.4})
        matching = module._mapped_joint_positions(
            {"name": ["shoulder_lift", "unmodeled_gripper"], "position": [0.2, 0.8]},
            {},
            {"shoulder_lift"},
        )
        self.assertEqual(matching, {"shoulder_lift": 0.2})
        with self.assertRaisesRegex(ValueError, "multiple ROS joints map"):
            module._mapped_joint_positions(
                {"name": ["left", "right"], "position": [0.1, 0.2]},
                {"left": "elbow_flex", "right": "elbow_flex"},
                {"elbow_flex"},
            )
        self.assertEqual(module._bridge_direction("ros_to_newton"), "ros_to_newton")
        with self.assertRaisesRegex(ValueError, "unsupported Newton ROS bridge direction"):
            module._bridge_direction("unsafe_loop")
        self.assertAlmostEqual(
            module._message_stamp_seconds({
                "header": {"stamp": {"sec": 1_700_000_000, "nanosec": 250_000_000}}
            }),
            1_700_000_000.25,
        )
        bridge = object.__new__(module.RosbridgeSession)
        bridge.joint_map = {"real_shoulder": "shoulder_lift"}
        bridge.allowed_joints = {"shoulder_lift"}
        bridge.run_id = "digital-twin-test"
        bridge.bridge_id = "test-bridge"
        bridge.command_stale_seconds = 0.5
        bridge.received = 0
        bridge.rejected = 0
        bridge.last_command_at = 0.0
        bridge.last_error = ""
        joint_state = {
            "header": {"stamp": {"sec": 1_700_000_000, "nanosec": 250_000_000}},
            "name": ["real_shoulder"],
            "position": [0.4],
        }
        with (
            mock.patch.object(module.runtime, "record_joint_observation") as record_observation,
            mock.patch.object(module.runtime, "command_session") as command_session,
        ):
            bridge._on_command(joint_state)
        record_observation.assert_called_once_with(
            "digital-twin-test",
            {"shoulder_lift": 0.4},
            source="rosbridge:test-bridge",
            observed_at=1_700_000_000.25,
            stale_after_seconds=0.5,
        )
        command_session.assert_called_once_with(
            "digital-twin-test", {"shoulder_lift": 0.4}, source="rosbridge:test-bridge"
        )
        self.assertEqual(bridge.received, 1)
        self.assertEqual(bridge.rejected, 0)
        bridge.received = 0
        bridge.rejected = 0
        with (
            mock.patch.object(module.runtime, "record_joint_observation") as record_observation,
            mock.patch.object(
                module.runtime,
                "command_session",
                side_effect=RuntimeError("simulation motion is disarmed"),
            ),
        ):
            bridge._on_command(joint_state)
        record_observation.assert_called_once()
        self.assertEqual(bridge.received, 1)
        self.assertEqual(bridge.rejected, 1)
        self.assertIn("disarmed", bridge.last_error)
        failed = module.newton_rosbridge({
            "action": "start",
            "bridge_id": "missing-session-bridge",
            "run_id": "missing-newton-session",
            "direction": "ros_to_newton",
        })
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["bridge"]["direction"], "ros_to_newton")
        self.assertIn("must be running first", failed["bridge"]["last_error"])
        source = bridge_path.read_text(encoding="utf-8")
        self.assertIn('"session": Dict(default={})', source)
        self.assertIn('"joint_map": Dict(default={})', source)
        self.assertIn('"ros_to_newton", "newton_to_ros"', source)

    def test_dataset_replay_maps_episode_frames_through_the_newton_safety_gate(self) -> None:
        package = load_package(
            PACKAGE_ROOT,
            component_overrides={"viewer-ovrtx": False, "rosbridge": False, "replay": True},
        )
        self.assertTrue(package.ok, package.error)
        self.assertIn("replay", package.enabled_components)
        self.assertIn("NewtonReplayBridge", package.node_types)

        bridge_path = PACKAGE_ROOT / "components" / "replay" / "nodes" / "bridge.py"
        spec = importlib.util.spec_from_file_location("blacknode_newton_replay_test", bridge_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        mapped = module._mapped_replay_positions(
            {
                "joint_names": ["recorded_shoulder", "camera_tilt"],
                "positions": [90.0, 12.0],
                "units": "degrees",
            },
            {"recorded_shoulder": "shoulder_lift"},
            {"shoulder_lift", "elbow_flex"},
            {"shoulder_lift": "radians"},
        )
        self.assertAlmostEqual(mapped["shoulder_lift"], math.pi / 2.0)
        self.assertEqual(
            module._mapped_replay_positions(
                {
                    "joint_names": ["shoulder_lift", "unmodeled_gripper"],
                    "positions": [0.2, 0.8],
                    "units": "radians",
                },
                {},
                {"shoulder_lift"},
                {"shoulder_lift": "radians"},
            ),
            {"shoulder_lift": 0.2},
        )
        with self.assertRaisesRegex(ValueError, "multiple dataset joints map"):
            module._mapped_replay_positions(
                {
                    "joint_names": ["left", "right"],
                    "positions": [0.1, 0.2],
                    "units": "radians",
                },
                {"left": "elbow_flex", "right": "elbow_flex"},
                {"elbow_flex"},
                {"elbow_flex": "radians"},
            )
        self.assertEqual(
            module._mapped_replay_positions(
                {"kind": "blacknode.stream-schema"}, {}, {"shoulder_lift"}, {}
            ),
            {},
        )
        with self.assertRaisesRegex(ValueError, "ws:// or wss://"):
            module._stream_url("http://127.0.0.1:8765")

        bridge = module.ReplayBridgeSession.__new__(module.ReplayBridgeSession)
        bridge.joint_map = {"recorded_shoulder": "shoulder_lift"}
        bridge.allowed_joints = {"shoulder_lift"}
        bridge.joint_units = {"shoulder_lift": "radians"}
        bridge.units = "auto"
        bridge.apply_seek = True
        bridge.bridge_id = "episode-replay-bridge"
        bridge.run_id = "episode-replay-test"
        bridge.received = 0
        bridge.applied = 0
        bridge.rejected = 0
        bridge.seek_frames = 0
        bridge.last_frame_index = -1
        bridge.last_command_at = 0.0
        bridge.command_stale_seconds = 0.5
        bridge.last_error = ""
        frame = {
            "joint_names": ["recorded_shoulder"],
            "positions": [30.0],
            "units": "degrees",
            "frame_index": 12,
            "playback_event": "seek",
        }
        with (
            mock.patch.object(module.runtime, "record_joint_observation") as record_observation,
            mock.patch.object(module.runtime, "command_session") as command_session,
        ):
            bridge._on_message(frame)
        record_observation.assert_called_once()
        observation_args, observation_kwargs = record_observation.call_args
        self.assertEqual(observation_args[0], "episode-replay-test")
        self.assertAlmostEqual(observation_args[1]["shoulder_lift"], math.pi / 6.0)
        self.assertEqual(
            observation_kwargs,
            {"source": "dataset-replay:episode-replay-bridge", "stale_after_seconds": 0.5},
        )
        command_session.assert_called_once()
        args, kwargs = command_session.call_args
        self.assertEqual(args[0], "episode-replay-test")
        self.assertAlmostEqual(args[1]["shoulder_lift"], math.pi / 6.0)
        self.assertEqual(kwargs, {"source": "dataset-replay"})
        self.assertEqual(bridge.applied, 1)
        self.assertEqual(bridge.seek_frames, 1)
        self.assertEqual(bridge.last_frame_index, 12)

        bridge.last_command_at = time.time() - 1.0
        bridge.stale_disarms = 0
        with (
            mock.patch.object(
                module.runtime,
                "session_status",
                return_value={"armed": True, "last_command_source": "dataset-replay"},
            ),
            mock.patch.object(module.runtime, "control_session") as control_session,
        ):
            bridge._check_stale()
        control_session.assert_called_once_with("episode-replay-test", "disarm")
        self.assertEqual(bridge.stale_disarms, 1)
        self.assertEqual(bridge.last_command_at, 0.0)

        failed = module.newton_replay_bridge({
            "action": "start",
            "bridge_id": "missing-session-replay",
            "run_id": "missing-newton-session",
            "stream_url": "ws://127.0.0.1:8765",
        })
        self.assertFalse(failed["ok"])
        self.assertIn("must be running first", failed["bridge"]["last_error"])

    def test_ovrtx_pose_conversion_preserves_newton_translation_and_box_scale(self) -> None:
        worker_path = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "worker.py"
        )
        spec = importlib.util.spec_from_file_location("blacknode_newton_ovrtx_worker_test", worker_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        matrix = module._pose_matrix(
            [0.25, -0.1, 0.04, 0.0, 0.0, 0.0, 1.0],
            [0.04, 0.05, 0.06],
        )
        self.assertEqual(matrix[0], 0.04)
        self.assertEqual(matrix[5], 0.05)
        self.assertEqual(matrix[10], 0.06)
        self.assertEqual(matrix[12:16], [0.25, -0.1, 0.04, 1])

        centimetre_matrix = module._pose_matrix(
            [0.25, -0.1, 0.04, 0.0, 0.0, 0.0, 1.0],
            [0.04, 0.05, 0.06],
            meters_per_unit=0.01,
            scale_in_meters=True,
        )
        self.assertEqual(centimetre_matrix[0:11:5], [4.0, 5.0, 6.0])
        self.assertEqual(centimetre_matrix[12:16], [25.0, -10.0, 4.0, 1])

        local_matrix = module._body_pose_matrix(
            {"index": 1, "parent_index": 0, "scale": [1.0, 1.0, 1.0]},
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 6.0, 8.0, 0.0, 0.0, 0.0, 1.0],
            ],
            1.0,
        )
        self.assertEqual(local_matrix[12:16], [3.0, 4.0, 5.0, 1.0])

        scaled_parent_matrix = module._body_pose_matrix(
            {
                "index": 0,
                "parent_index": -1,
                "scale": [1.0, 1.0, 1.0],
                "world_scale": [0.01, 0.01, 0.01],
                "render_parent_index": -1,
                "render_parent_relative": [
                    0.01, 0.0, 0.0, 0.0,
                    0.0, 0.01, 0.0, 0.0,
                    0.0, 0.0, 0.01, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
            },
            [[0.321, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            1.0,
        )
        self.assertAlmostEqual(scaled_parent_matrix[0], 1.0)
        self.assertAlmostEqual(scaled_parent_matrix[5], 1.0)
        self.assertAlmostEqual(scaled_parent_matrix[10], 1.0)
        self.assertAlmostEqual(scaled_parent_matrix[12], 32.1)

        collider_matrix = module._bound_shape_pose_matrix(
            {
                "body_index": 0,
                "initial_world_matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    1.25, 2.0, 3.0, 1.0,
                ],
                "body_bind_world_matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    1.0, 2.0, 3.0, 1.0,
                ],
            },
            [[2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0]],
            1.0,
        )
        self.assertEqual(collider_matrix[12:16], [2.25, 4.0, 6.0, 1.0])
        nested_matrix = module._bound_shape_pose_matrix(
            {
                "body_index": 0,
                "initial_world_matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    1.25, 2.0, 3.0, 1.0,
                ],
                "body_bind_world_matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    1.0, 2.0, 3.0, 1.0,
                ],
                "render_parent_world_matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    2.0, 3.0, 4.0, 1.0,
                ],
            },
            [[2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0]],
            1.0,
        )
        for actual, expected in zip(nested_matrix[12:16], [0.25, 1.0, 2.0, 1.0]):
            self.assertAlmostEqual(actual, expected)
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        moved = list(identity)
        moved[12] = 1.0
        delta = module._interaction_transform_delta(
            "/World/Cube/Visual",
            {"/World/Cube/Visual": {
                "local_matrix": identity,
                "parent_world": identity,
            }},
            {"/World/Cube/Visual": moved},
        )
        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta[12], 1.0)
        self.assertTrue(module._transforms_are_finite([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]))
        self.assertFalse(module._transforms_are_finite([
            [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]))

        payload = module._matrix_array([matrix])
        self.assertEqual(str(payload.dtype), "float64")
        self.assertEqual(payload.shape, (16,))
        self.assertEqual(module._encode_frame({}, 90), b"")

        from PIL import Image

        semantic_ids = __import__("numpy").zeros((9, 9), dtype="uint32")
        semantic_ids[2:7, 2:7] = 7
        selected_image = module._draw_selection_outline(
            Image.new("RGB", (9, 9), (20, 30, 40)),
            semantic_ids,
            {7: "label:SelectionCube"},
            "/World/SelectionCube",
        )
        self.assertEqual(selected_image.getpixel((1, 4)), (213, 169, 53))
        self.assertEqual(selected_image.getpixel((4, 4)), (20, 30, 40))
        wire_camera = module.Camera({
            "camera": {
                "position": [0.0, -10.0, 0.0],
                "target": [0.0, 0.0, 0.0],
                "up_axis": "z",
            }
        })
        wire_image = module._draw_collision_wireframes(
            Image.new("RGB", (100, 100), (0, 0, 0)),
            [{
                "points_bind_world": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "edges": [[0, 1]],
                "body_index": -1,
                "body_bind_world_matrix": [],
            }],
            [],
            wire_camera,
            1.0,
        )
        wire_pixels = __import__("numpy").asarray(wire_image)
        self.assertTrue(__import__("numpy").any(wire_pixels[:, :, 1] > 200))

        provider_path = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
        )
        provider_source = provider_path.read_text(encoding="utf-8")
        self.assertIn('draggable="false"', provider_source)
        self.assertIn("requestAnimationFrame", provider_source)
        self.assertIn("blacknode-newton-selection", provider_source)
        self.assertIn("blacknode-newton-transform", provider_source)
        self.assertIn("/api/transform-preview", provider_source)
        self.assertIn("opacity:0", provider_source)
        self.assertIn("blacknode-newton-view-state", provider_source)
        self.assertIn("event.source!==window.parent", provider_source)
        self.assertIn('data-tool="select"', provider_source)
        self.assertIn('data-tool="move"', provider_source)
        self.assertIn('data-tool="rotate"', provider_source)
        self.assertIn('data-tool="scale"', provider_source)
        self.assertIn("q:'select',w:'move',e:'rotate',r:'scale'", provider_source)
        self.assertIn("transform.rotate_deg", provider_source)
        self.assertIn("transform.scale", provider_source)
        self.assertIn("drag.b===0?'orbit':drag.b===1?'pan':'zoom'", provider_source)
        self.assertNotIn("Selecting…", provider_source)
        self.assertIn("_wait_for_selection", provider_source)
        provider_spec = importlib.util.spec_from_file_location(
            "blacknode_newton_ovrtx_provider_interaction_test", provider_path
        )
        self.assertIsNotNone(provider_spec)
        self.assertIsNotNone(provider_spec.loader)
        provider_module = importlib.util.module_from_spec(provider_spec)
        provider_spec.loader.exec_module(provider_module)
        self.assertEqual(provider_module._render_resolution({}), (1920, 1080))
        self.assertEqual(
            provider_module._render_resolution({"width": 4096, "height": 100}),
            (3840, 240),
        )
        self.assertEqual(
            provider_module._workspace_scene_path("/BlacknodeOVRT/Ground/Geometry"),
            "/Blacknode/Ground",
        )
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as directory:
            interaction_asset = Path(directory) / "interaction.usda"
            interaction_stage = Usd.Stage.CreateNew(str(interaction_asset))
            root = UsdGeom.Xform.Define(interaction_stage, "/World")
            root.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
            box = UsdGeom.Cube.Define(interaction_stage, "/World/Box")
            box.AddTranslateOp().Set(Gf.Vec3d(32.1, 0.0, 0.0))
            interaction_stage.GetRootLayer().Save()
            frames = provider_module._usd_interaction_frames(
                str(interaction_asset),
                [{"path": "/World/Box", "editable": True}],
            )
            self.assertEqual(len(frames), 1)
            self.assertAlmostEqual(frames[0]["parent_world"][0], 0.01)
            self.assertAlmostEqual(frames[0]["local_matrix"][12], 32.1)
        zoom_drag = "(pending?.delta||0)-(dx+dy)*Math.SQRT1_2"
        self.assertIn(zoom_drag, provider_source)
        self.assertIn(zoom_drag, worker_path.read_text(encoding="utf-8"))

        camera = module.Camera({
            "camera": {"position": [0.0, -10.0, 0.0], "target": [0.0, 0.0, 0.0], "up_axis": "z"}
        })
        camera.apply([{"action": "zoom", "delta": -10.0}])
        self.assertLess(float(__import__("numpy").linalg.norm(camera.eye - camera.target)), 10.0)
        camera.apply([{"action": "reset"}, {"action": "zoom", "delta": 10.0}])
        self.assertGreater(float(__import__("numpy").linalg.norm(camera.eye - camera.target)), 10.0)
        self.assertTrue(camera.apply([{
            "action": "set",
            "position": [1.0, 2.0, 3.0],
            "target": [0.0, 0.0, 0.0],
            "up_vector": [1.0, 0.0, 0.0],
        }]))
        self.assertEqual(camera.eye.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(camera.target.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(camera.up_vector.tolist(), [1.0, 0.0, 0.0])
        self.assertFalse(module._requires_renderer_reset([], []))
        self.assertFalse(module._requires_renderer_reset(
            [{"type": "transform"}], []
        ))
        self.assertTrue(module._requires_renderer_reset(
            [{"type": "material"}], []
        ))
        self.assertTrue(module._requires_renderer_reset(
            [{"type": "environment"}], []
        ))
        self.assertTrue(module._requires_renderer_reset(
            [], [{"path": "/World/Box", "visible": False}]
        ))

        gizmo_camera = module.Camera({
            "camera": {"position": [4.0, -6.0, 3.0], "target": [0.0, 0.0, 0.0], "up_axis": "z"}
        })
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        gizmo = module._selection_gizmo(
            gizmo_camera,
            {"path": "/World/Box", "parent_world": identity, "local_matrix": identity},
            {"width": 1280, "height": 720, "meters_per_unit": 1.0},
            {},
        )
        self.assertAlmostEqual(gizmo["x"], 0.5)
        self.assertAlmostEqual(gizmo["y"], 0.5)
        self.assertEqual(set(gizmo["axes"]), {"x", "y", "z"})
        self.assertTrue(all(axis["pixels_per_meter"] > 0 for axis in gizmo["axes"].values()))
        self.assertEqual(set(gizmo["local_axes"]), {"x", "y", "z"})
        self.assertTrue(all(axis["pixels_per_meter"] > 0 for axis in gizmo["local_axes"].values()))
        self.assertTrue(all(
            len(axis.get("ring") or []) == 33 for axis in gizmo["local_axes"].values()
        ))
        native_matrices = module._native_gizmo_matrices(
            gizmo_camera,
            {"path": "/World/Box", "parent_world": identity, "local_matrix": identity},
            {"meters_per_unit": 1.0},
            {},
        )
        self.assertEqual(len(native_matrices), 3)
        self.assertEqual(native_matrices[0][12:15], [0.0, 0.0, 0.0])
        self.assertGreater(native_matrices[0][0], 0.0)

        class OutlineRenderer:
            def __init__(self) -> None:
                self.calls = []

            def set_selection_outline_group_strings(self, paths, groups) -> None:
                self.calls.append((list(paths), list(groups)))

        outline_renderer = OutlineRenderer()
        self.assertEqual(
            module._selection_outline_paths("/BlacknodeOVRT/Ground", []),
            ["/BlacknodeOVRT/Ground/Geometry"],
        )
        module._apply_selection_outline(
            outline_renderer, ["/World/Old"], ["/World/New", "/World/New/Mesh"]
        )
        self.assertEqual(outline_renderer.calls, [
            (["/World/Old"], [0]),
            (["/World/New", "/World/New/Mesh"], [1, 1]),
        ])

        class PathDictionary:
            def intern_path(self, path: str) -> int:
                return {"/World/Old": 10, "/World/New": 20}[path]

        class IdOutlineRenderer:
            def __init__(self) -> None:
                self.calls = []

            def set_selection_outline_group(self, path_ids, group) -> None:
                self.calls.append((list(path_ids), group))

        id_outline_renderer = IdOutlineRenderer()
        module._apply_selection_outline(
            id_outline_renderer,
            ["/World/Old"],
            ["/World/New"],
            PathDictionary(),
        )
        self.assertEqual(id_outline_renderer.calls, [([10], 0), ([20], 1)])
        module._apply_collider_outline(
            outline_renderer, ["/World/Collider"], True
        )
        module._apply_collider_outline(
            outline_renderer, ["/World/Collider"], False
        )
        self.assertEqual(outline_renderer.calls[-2:], [
            (["/World/Collider"], [2]),
            (["/World/Collider"], [0]),
        ])
        worker_source = worker_path.read_text(encoding="utf-8")
        self.assertIn("selection_outline_width=3", worker_source)
        self.assertIn(
            "selection_fill_mode=ovrtx.SelectionFillMode.EDGE_ONLY", worker_source
        )
        self.assertIn("renderer.reset()", worker_source)
        self.assertIn("outline_color=(1.0, 0.78, 0.22, 1.0)", worker_source)
        self.assertIn("fill_color=(1.0, 0.78, 0.22, 1.0)", worker_source)
        self.assertIn("outline_color=(0.20, 1.0, 0.32, 1.0)", worker_source)
        provider_source = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
        ).read_text(encoding="utf-8")
        self.assertIn('#g[data-tool="select"]{display:none!important}', provider_source)
        self.assertIn("if(!selection||tool==='select')", provider_source)
        self.assertNotIn('class="pivot"', provider_source)
        self.assertNotIn('class="name"', provider_source)

    def test_ovrtx_wrapper_preserves_usd_stage_semantics_and_authors_a_valid_grid(self) -> None:
        worker_path = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "worker.py"
        )
        spec = importlib.util.spec_from_file_location("blacknode_newton_ovrtx_wrapper_test", worker_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "scene@centimetres.usda"
            hdri = Path(directory) / "studio.exr"
            hdri.write_bytes(b"placeholder")
            asset.write_text(
                '''#usda 1.0\n(\n    upAxis = "Y"\n    metersPerUnit = 0.01\n)\n\ndef Cube "Box"\n{\n    double size = 100\n}\n''',
                encoding="utf-8",
            )
            config = {
                "asset_path": str(asset),
                "width": 640,
                "height": 360,
                "ground_height": 0.0,
                "ground_enabled": True,
                "show_grid": True,
                "meters_per_unit": 0.01,
                "up_axis": "y",
                "grid_extent": 100.0,
                "body_entries": [],
                "ghost_entries": [{
                    "index": 0,
                    "path": "/BlacknodeOVRT/RealGhost/shape_0",
                    "source_path": "/Box",
                    "scale": [1.0, 1.0, 1.0],
                }],
                "background_color": "#111827",
                "environment": {"hdri_path": str(hdri), "intensity": 2.0},
                "ground_transform": {
                    "translate_m": [1.0, 2.0, 3.0],
                    "rotate_deg": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "ground_material": {
                    "base_color": [0.1, 0.3, 0.8],
                    "metallic": 0.6,
                    "roughness": 0.25,
                    "opacity": 0.9,
                },
            }
            wrapper = module._wrapper_usda(
                config,
                module._camera_matrix([180.0, 135.0, 180.0], [0.0, 0.0, 0.0], "y"),
            )

        from pxr import Sdf

        layer = Sdf.Layer.CreateAnonymous(".usda")
        self.assertTrue(layer.ImportFromString(wrapper))
        self.assertIn("subLayers = [@@@", wrapper)
        self.assertIn("scene@centimetres.usda@@@", wrapper)
        self.assertIn('upAxis = "Y"', wrapper)
        self.assertIn("metersPerUnit = 0.01", wrapper)
        self.assertIn("float2 clippingRange = (1, 100000)", wrapper)
        self.assertIn("DistanceToCameraSD", wrapper)
        self.assertIn("SemanticSegmentation", wrapper)
        self.assertIn("SemanticIdMap", wrapper)
        self.assertIn("float inputs:intensity = 0", wrapper)
        self.assertIn("float inputs:intensity = 2500", wrapper)
        self.assertIn("color3f inputs:color", wrapper)
        self.assertIn("uint[] deviceIds = [0]", wrapper)
        self.assertIn("OmniRtxSettingsCommonAPI_1", wrapper)
        self.assertIn("int omni:rtx:background:source:type = 0", wrapper)
        self.assertIn("omni:rtx:background:source:texture:path", wrapper)
        self.assertNotIn("asset inputs:texture:file", wrapper)
        self.assertIn('interpolation = "constant"', wrapper)
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/Grid"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GridAxes"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GizmoMove/XTip"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GizmoRotate/Rings"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GizmoScale/XHandle"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/Ground"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/Ground/Geometry"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GroundMaterial"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GroundMaterial/PreviewSurface"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/RealGhost/shape_0"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/RealGhostMaterial/PreviewSurface"))
        self.assertIn("float inputs:opacity = 0.28", wrapper)
        self.assertEqual(module._render_prim_path("/Blacknode/Ground"), "/BlacknodeOVRT/Ground")
        self.assertIn("ovstage.population.open_usd_from_string(", worker_path.read_text(encoding="utf-8"))

        from pxr import Usd, UsdShade

        layer.subLayerPaths = []
        stage = Usd.Stage.Open(layer)
        sky_color = stage.GetPrimAtPath("/BlacknodeOVRT/Sky").GetAttribute("inputs:color").Get()
        self.assertEqual(
            [round(float(value), 6) for value in sky_color],
            [round(17 / 255, 6), round(24 / 255, 6), round(39 / 255, 6)],
        )
        ground = stage.GetPrimAtPath("/BlacknodeOVRT/Ground")
        translation = ground.GetAttribute("xformOp:transform").Get().ExtractTranslation()
        self.assertEqual([float(value) for value in translation], [100.0, 200.0, 300.0])
        bound = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/BlacknodeOVRT/Ground/Geometry")
        ).ComputeBoundMaterial()[0]
        self.assertEqual(str(bound.GetPath()), "/BlacknodeOVRT/GroundMaterial")
        shader = UsdShade.Shader(stage.GetPrimAtPath("/BlacknodeOVRT/GroundMaterial/PreviewSurface"))
        self.assertAlmostEqual(float(shader.GetInput("metallic").Get()), 0.6, places=6)
        self.assertAlmostEqual(float(shader.GetInput("roughness").Get()), 0.25, places=6)

        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            decoded_exr = Path(directory) / "decoded.exr"
            decoded_exr.write_bytes(b"test")
            imageio_frame = np.asarray(
                [[[[255, 64, 0], [0, 128, 255]], [[32, 64, 96], [255, 255, 255]]]],
                dtype=np.uint8,
            )
            with mock.patch("imageio.v3.imread", return_value=imageio_frame):
                environment_image = module._load_environment_image(str(decoded_exr))
            self.assertEqual(environment_image.shape, (2, 2, 3))
            self.assertEqual(environment_image.dtype, np.float32)
            self.assertAlmostEqual(float(environment_image[0, 0, 0]), 1.0)
            environment_camera = module.Camera({
                "camera": {
                    "position": [2.0, -2.0, 1.0],
                    "target": [0.0, 0.0, 0.0],
                    "up_axis": "z",
                }
            })
            projected_environment = module._project_environment_background(
                environment_camera,
                environment_image,
                {"width": 16, "height": 8},
                {"show_background": True, "intensity": 1.0},
            )
            self.assertEqual(projected_environment.shape, (8, 16, 3))
            hidden_environment = module._project_environment_background(
                environment_camera,
                environment_image,
                {"width": 16, "height": 8},
                {
                    "show_background": False,
                    "intensity": 1.0,
                    "background_color": "#6383c5",
                },
            )
            self.assertEqual(hidden_environment.shape, (8, 16, 3))
            self.assertEqual(hidden_environment[0, 0].tolist(), [99, 131, 197])
            interactive_camera = mock.Mock()
            interactive_camera.environment_background.return_value = np.zeros(
                (360, 640, 3), dtype=np.uint8
            )
            interactive_background = module._project_environment_background(
                interactive_camera,
                environment_image,
                {"width": 1920, "height": 1080},
                {"show_background": True, "intensity": 1.0},
                interactive=True,
            )
            self.assertEqual(interactive_background.shape, (1080, 1920, 3))
            interactive_camera.environment_background.assert_called_once_with(
                environment_image, 640, 360, 1.0
            )
            self.assertIsNone(module._try_load_environment_image(str(Path(directory) / "missing.exr")))

            from PIL import Image

            png_output = io.BytesIO()
            Image.fromarray(np.asarray([[[255, 64, 0], [0, 128, 255]]], dtype=np.uint8)).save(
                png_output, format="PNG"
            )
            ffmpeg_result = mock.Mock(returncode=0, stdout=png_output.getvalue(), stderr=b"")
            fake_ffmpeg = mock.Mock()
            fake_ffmpeg.get_ffmpeg_exe.return_value = "ffmpeg"
            with (
                mock.patch("imageio.v3.imread", side_effect=OSError("no pyav")),
                mock.patch("cv2.imread", return_value=None),
                mock.patch.dict(sys.modules, {"imageio_ffmpeg": fake_ffmpeg}),
                mock.patch("subprocess.run", return_value=ffmpeg_result) as ffmpeg_run,
            ):
                ffmpeg_image = module._load_environment_image(str(decoded_exr))
            self.assertEqual(ffmpeg_image.shape, (1, 2, 3))
            self.assertEqual(ffmpeg_image.dtype, np.float32)
            self.assertTrue(ffmpeg_run.called)
            self.assertNotIn(
                "scale=2048:2048:force_original_aspect_ratio=decrease",
                ffmpeg_run.call_args.args[0],
            )

        preset_path = module._environment_texture_path({"hdri": "studio"})
        self.assertTrue(Path(preset_path).is_file(), preset_path)
        self.assertEqual(Path(preset_path).name, "studio_small_03_1k.jpg")
        preset_wrapper = module._wrapper_usda({
            "asset_path": "", "width": 320, "height": 240,
            "ground_height": 0.0, "ground_enabled": False, "show_grid": False,
            "meters_per_unit": 1.0, "up_axis": "z", "grid_extent": 1.0,
            "body_entries": [], "background_color": "#6383c5",
            "environment": {"hdri": "studio", "show_background": True, "intensity": 0.0},
        }, module._camera_matrix([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], "z"))
        self.assertIn("studio_small_03_1k.jpg", preset_wrapper)
        self.assertIn("int omni:rtx:background:source:type = 0", preset_wrapper)
        self.assertEqual(preset_wrapper.count("float inputs:intensity = 0"), 1)
        self.assertIn("float inputs:intensity = 2500", preset_wrapper)
        self.assertEqual(module._dome_light_intensity({
            "hdri": "studio", "intensity": 1.5,
        }), 1500.0)
        self.assertEqual(module._dome_light_intensity({
            "hdri": "none", "intensity": 1.5,
        }), 675.0)
        managed_dome = module._environment_dome_usda(
            preset_path, {"hdri": "studio", "intensity": 2.0}
        )
        self.assertIn('defaultPrim = "EnvironmentDome"', managed_dome)
        self.assertIn("asset inputs:texture:file", managed_dome)
        self.assertIn("studio_small_03_1k.jpg", managed_dome)
        self.assertIn("float inputs:intensity = 2000", managed_dome)
        self.assertIn("bool inputs:visibleInPrimaryRay = false", managed_dome)
        fake_ovstage = mock.Mock()
        fake_ovstage.population.add_usd_reference_from_string.return_value = 23
        fake_stage = object()
        with mock.patch.dict(sys.modules, {"ovstage": fake_ovstage}):
            replaced_handle = module._replace_environment_dome(
                fake_stage,
                17,
                preset_path,
                {"hdri": "studio", "intensity": 2.0},
                9,
            )
        self.assertEqual(replaced_handle, 23)
        fake_ovstage.population.remove_usd.assert_called_once_with(fake_stage, 17)
        fake_ovstage.population.add_usd_reference_from_string.assert_called_once_with(
            fake_stage,
            mock.ANY,
            "/BlacknodeOVRT/EnvironmentDome",
        )
        fake_ovstage.population.apply_usd_changes.assert_called_once_with(fake_stage, 9)
        self.assertEqual(module._background_source_type({
            "hdri": "studio", "show_background": True,
        }), 0)
        self.assertEqual(module._background_source_type({
            "hdri": "studio", "show_background": False,
        }), 2)
        key_wrapper = module._wrapper_usda({
            "asset_path": "", "width": 320, "height": 240,
            "ground_height": 0.0, "ground_enabled": False, "show_grid": False,
            "meters_per_unit": 1.0, "up_axis": "z", "grid_extent": 1.0,
            "body_entries": [], "background_color": "#6383c5",
            "environment": {
                "hdri": "none",
                "distant_light": {
                    "enabled": False,
                    "intensity": 1250,
                    "color": "#3366cc",
                    "angle_deg": 8,
                    "rotation_deg": [-20, 15, 40],
                },
            },
        }, module._camera_matrix([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], "z"))
        self.assertIn("float inputs:intensity = 0", key_wrapper)
        self.assertIn(
            "color3f inputs:color = (0.0331048, 0.132868, 0.603827)",
            key_wrapper,
        )
        self.assertIn("float inputs:angle = 8", key_wrapper)
        self.assertIn("float3 xformOp:rotateXYZ = (-20, 15, 40)", key_wrapper)
        middle_gray = module._distant_light({
            "distant_light": {"color": "#808080"}
        })["color"]
        self.assertAlmostEqual(middle_gray[0], 0.215861, places=5)
        self.assertEqual(middle_gray[0], middle_gray[1])
        background_source_tensor = module._background_source_tensor({
            "hdri": "studio", "show_background": True,
        })
        self.assertEqual(str(background_source_tensor.dtype), "uint64")
        self.assertEqual(background_source_tensor.tolist(), [0])

        infrared = module._infrared_depth(np.asarray([[[1.0], [2.0]], [[3.0], [float("inf")]]]))
        self.assertEqual(infrared.shape, (2, 2, 3))
        self.assertEqual(infrared[1, 1].tolist(), [0, 0, 0])
        self.assertTrue(module._render_shape_visible(
            {"visual": True, "collider": False}, True, False
        ))
        self.assertFalse(module._render_shape_visible(
            {"visual": True, "collider": False}, False, True
        ))
        self.assertFalse(module._render_shape_visible(
            {"visual": False, "collider": True}, False, True
        ))
        self.assertFalse(module._render_shape_visible(
            {"path": "/World/Robot/Visual", "visual": True, "collider": False},
            True,
            False,
            {"/World/Robot": False},
        ))
        self.assertFalse(module._render_shape_visible(
            {"path": "/World/Collider", "visual": False, "collider": True},
            False,
            True,
            {"/World/Collider": False},
        ))
        self.assertFalse(module._render_shape_visible(
            {
                "path": "/__BlacknodeColliderDisplay/Geometry/collider_1",
                "source_path": "/World/Table/Top",
                "visual": False,
                "collider": True,
            },
            False,
            True,
            {"/World/Table": False},
        ))
        segments, boxes = module._semantic_visuals(
            np.asarray([[0, 2, 2, 0], [0, 2, 2, 0]] * 4, dtype=np.uint32),
            {2: "class: cube; label: Box;"},
        )
        self.assertEqual(segments.shape, (8, 4, 3))
        self.assertEqual(boxes[0]["label"], "Box")
        self.assertEqual(boxes[0]["bounds"], (1, 0, 2, 7))

    def test_workspace_usd_overrides_are_non_destructive_and_inspectable(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "editable.usda"
            stage = Usd.Stage.CreateNew(str(asset))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 0.25)
            UsdGeom.Xform.Define(stage, "/World")
            UsdGeom.Cube.Define(stage, "/World/Box")
            colored = UsdGeom.Cube.Define(stage, "/World/Colored")
            colored.CreateDisplayColorAttr([(0.2, 0.4, 0.6)])
            physical = UsdGeom.Cube.Define(stage, "/World/Physical")
            UsdPhysics.CollisionAPI.Apply(physical.GetPrim())
            proxy = UsdGeom.Cube.Define(stage, "/World/Proxy")
            proxy.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
            UsdGeom.Xform.Define(stage, "/World/Collisions")
            UsdGeom.Cube.Define(stage, "/World/Collisions/Collider")
            stage.GetRootLayer().Save()
            source_text = asset.read_text(encoding="utf-8")
            composed, overlay_path, items = runtime._compose_workspace_stage(
                str(asset),
                {
                    "visibility": {"/World/Box": False},
                    "transforms": {
                        "/World/Box": {
                            "translate_m": [1.0, 2.0, 3.0],
                            "rotate_deg": [10.0, 20.0, 30.0],
                            "scale": [1.0, 2.0, 1.0],
                        }
                    },
                    "materials": {
                        "/World/Box": {
                            "base_color": [0.1, 0.2, 0.3],
                            "metallic": 0.4,
                            "roughness": 0.6,
                            "opacity": 0.8,
                        }
                    },
                },
            )
            try:
                item = next(value for value in items if value["path"] == "/World/Box")
                colored_item = next(value for value in items if value["path"] == "/World/Colored")
                physical_item = next(
                    value for value in items if value["path"] == "/World/Physical"
                )
                proxy_item = next(
                    value for value in items if value["path"] == "/World/Proxy"
                )
                collider_item = next(
                    value for value in items
                    if value["path"] == "/World/Collisions/Collider"
                )
                self.assertFalse(item["visible"])
                self.assertEqual(physical_item["render_role"], "collider")
                self.assertFalse(physical_item["collision_only"])
                self.assertEqual(proxy_item["render_role"], "collider")
                self.assertTrue(proxy_item["collision_only"])
                self.assertEqual(collider_item["render_role"], "collider")
                self.assertTrue(collider_item["collision_only"])
                collision_proxy = composed.GetPrimAtPath(
                    runtime._collision_display_path("/World/Physical")
                )
                self.assertTrue(collision_proxy and collision_proxy.IsValid())
                self.assertEqual(
                    collision_proxy.GetAttribute("blacknode:sourcePath").Get(),
                    "/World/Physical",
                )
                self.assertTrue(
                    UsdShade.MaterialBindingAPI(collision_proxy).ComputeBoundMaterial()[0]
                )
                self.assertFalse(
                    UsdGeom.Gprim(composed.GetPrimAtPath("/World/Physical"))
                    .GetDisplayColorAttr()
                    .HasAuthoredValueOpinion()
                )
                for collision_path in ("/World/Proxy", "/World/Collisions/Collider"):
                    collision_gprim = UsdGeom.Gprim(
                        composed.GetPrimAtPath(collision_path)
                    )
                    self.assertTrue(
                        collision_gprim.GetDisplayColorAttr().HasAuthoredValueOpinion()
                    )
                    self.assertAlmostEqual(
                        float(collision_gprim.GetDisplayOpacityAttr().Get()[0]), 0.42
                    )
                self.assertEqual(item["transform"]["translate_m"], [1.0, 2.0, 3.0])
                for actual, expected in zip(
                    colored_item["material"]["base_color"], [0.2, 0.4, 0.6], strict=True
                ):
                    self.assertAlmostEqual(actual, expected, places=6)
                bound = UsdShade.MaterialBindingAPI(composed.GetPrimAtPath("/World/Box")).ComputeBoundMaterial()[0]
                self.assertTrue(bound)
                colored_keeper = composed.GetPrimAtPath(
                    runtime._workspace_material_keeper_path("/World/Colored")
                )
                self.assertTrue(colored_keeper and colored_keeper.IsValid())
                self.assertTrue(colored_keeper.IsA(UsdGeom.Mesh))
                self.assertEqual(
                    UsdGeom.Imageable(colored_keeper).ComputeVisibility(),
                    UsdGeom.Tokens.invisible,
                )
                self.assertFalse(
                    UsdGeom.Mesh(colored_keeper).GetPointsAttr().HasAuthoredValueOpinion()
                )
                kept_material = UsdShade.MaterialBindingAPI(
                    colored_keeper
                ).ComputeBoundMaterial()[0]
                self.assertEqual(
                    str(kept_material.GetPath()),
                    runtime._workspace_material_path("/World/Colored"),
                )
                box_prim = composed.GetPrimAtPath("/World/Box")
                self.assertTrue(box_prim.HasAPI(UsdShade.MaterialBindingAPI))
                self.assertTrue(
                    composed.GetPrimAtPath("/World/Physical").HasAPI(
                        UsdShade.MaterialBindingAPI
                    )
                )
                self.assertIn("SemanticsAPI:class", str(box_prim.GetMetadata("apiSchemas")))
                self.assertEqual(
                    box_prim.GetAttribute("semantic:label:params:semanticData").Get(), "Box"
                )
                self.assertTrue(Path(overlay_path).is_file())
                self.assertEqual(UsdGeom.GetStageUpAxis(composed), UsdGeom.Tokens.z)
                self.assertEqual(UsdGeom.GetStageMetersPerUnit(composed), 0.25)
                exported = Usd.Stage.Open(overlay_path)
                self.assertEqual(UsdGeom.GetStageUpAxis(exported), UsdGeom.Tokens.z)
                self.assertEqual(UsdGeom.GetStageMetersPerUnit(exported), 0.25)
                self.assertEqual(asset.read_text(encoding="utf-8"), source_text)
            finally:
                Path(overlay_path).unlink(missing_ok=True)

    def test_workspace_visibility_updates_the_existing_viewer_and_scene_items(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        class LiveViewer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def set_visibility(self, path: str, visible: bool) -> bool:
                self.calls.append((path, visible))
                return True

        session = object.__new__(runtime.NewtonSession)
        session.viewer = LiveViewer()
        session.viewer_config = {"provider": "ovrtx"}
        session.lock = __import__("threading").RLock()
        session.scene_items = [
            {"path": "/World", "visible": True},
            {"path": "/World/Robot", "visible": True},
            {"path": "/World/Robot/Arm", "visible": True},
        ]
        session.status = lambda: {"scene_items": session.scene_items}

        result = session.set_visibility(
            "/World/Robot", False, {"/World/Robot": False}
        )

        self.assertEqual(session.viewer.calls, [("/World/Robot", False)])
        self.assertTrue(result["scene_items"][0]["visible"])
        self.assertFalse(result["scene_items"][1]["visible"])
        self.assertFalse(result["scene_items"][2]["visible"])
        workspace_source = __import__("inspect").getsource(runtime.control_workspace)
        visibility_branch = workspace_source.split('if command == "set_visibility":', 1)[1].split(
            'if command == "set_transform":', 1
        )[0]
        self.assertNotIn("_restart_workspace", visibility_branch)

    def test_dynamic_transform_updates_newton_spawn_pose_and_pauses_physics(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime
        import warp as wp

        class LiveViewer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []

            def set_transform(self, path: str, transform: dict[str, object]) -> bool:
                self.calls.append((path, transform))
                return True

        class State:
            def __init__(self) -> None:
                self.body_q = wp.array(
                    [[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]],
                    dtype=wp.transform,
                    device="cpu",
                )
                self.body_qd = wp.array(
                    [[3.0, 2.0, 1.0, 0.5, 0.25, 0.125]],
                    dtype=wp.spatial_vector,
                    device="cpu",
                )

        session = object.__new__(runtime.NewtonSession)
        session.viewer = LiveViewer()
        session.viewer_config = {"provider": "ovrtx"}
        session.lock = __import__("threading").RLock()
        session.model = type(
            "Model", (), {"body_label": ["/World/Cube"], "device": "cpu"}
        )()
        session.dynamic_body_indices = {0}
        session.body_pose_overrides = {}
        session.scene_items = [{
            "path": "/World/Cube",
            "parent_path": "/",
            "transform": {
                "translate_m": [1.0, 0.0, 2.0],
                "rotate_deg": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }]
        session.state_0 = State()
        session.state_1 = State()
        session.paused = False
        session.phase = "running"
        session._imports = lambda: (object(), wp)
        session.status = lambda: {
            "paused": session.paused,
            "phase": session.phase,
            "pose": session.state_0.body_q.numpy().tolist()[0],
        }

        result = session.set_transform("/World/Cube", {
            "translate_m": [2.0, 0.0, 3.0],
            "rotate_deg": [0.0, 0.0, 90.0],
            "scale": [1.0, 1.0, 1.0],
        })

        self.assertTrue(result["paused"])
        self.assertEqual(result["phase"], "paused")
        self.assertEqual(
            [round(float(value), 5) for value in result["pose"][:3]],
            [2.0, 0.0, 3.0],
        )
        self.assertAlmostEqual(abs(float(result["pose"][5])), math.sqrt(0.5), places=5)
        self.assertAlmostEqual(abs(float(result["pose"][6])), math.sqrt(0.5), places=5)
        self.assertEqual(
            session.state_0.body_qd.numpy().tolist()[0], [0.0] * 6
        )
        self.assertTrue(session.viewer.calls[0][1]["_physics_backed"])
        for expected, actual in zip(session.body_pose_overrides[0], result["pose"]):
            self.assertAlmostEqual(float(expected), float(actual), places=5)

    def test_scene_items_resolve_mesh_picks_to_their_physics_xform(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        session = object.__new__(runtime.NewtonSession)
        session.model = type(
            "Model", (), {"body_label": ["/World/Robot", "/World/Cube"]}
        )()
        session.dynamic_body_indices = {1}
        session.scene_items = [
            {"path": "/World/Robot/Visual", "editable": True},
            {"path": "/World/Cube", "editable": True},
            {"path": "/World/Cube/Visual", "editable": True},
        ]

        session._annotate_scene_item_physics()

        robot, cube_xform, cube = session.scene_items
        self.assertTrue(robot["editable"])
        self.assertFalse(robot["physics_dynamic"])
        self.assertEqual(robot["physics_body_path"], "/World/Robot")
        self.assertFalse(robot["physics_pose_editable"])
        self.assertTrue(cube["physics_dynamic"])
        self.assertEqual(cube["physics_body_index"], 1)
        self.assertEqual(cube["physics_body_path"], "/World/Cube")
        self.assertFalse(cube["physics_pose_editable"])
        self.assertTrue(cube_xform["physics_pose_editable"])

    def test_usd_collision_guides_are_separate_live_render_geometry(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        scaled_body = Gf.Transform()
        scaled_body.SetScale(Gf.Vec3d(0.01, 0.01, 0.01))
        scaled_body.SetTranslation(Gf.Vec3d(1.0, 2.0, 3.0))
        rigid_body = Gf.Transform(
            runtime._usd_rigid_matrix(scaled_body.GetMatrix(), Gf)
        )
        self.assertEqual(
            [round(float(value), 8) for value in rigid_body.GetScale()],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            [float(value) for value in rigid_body.GetTranslation()],
            [1.0, 2.0, 3.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "collision-guides.usda"
            stage = Usd.Stage.CreateNew(str(asset))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.Xform.Define(stage, "/World")
            UsdGeom.Cube.Define(stage, "/World/Visual")
            shared = UsdGeom.Mesh.Define(stage, "/World/Shared")
            shared.CreatePointsAttr([
                (0.0, 0.0, 0.0), (0.1, 0.0, 0.0),
                (0.0, 0.1, 0.0), (0.0, 0.0, 0.1),
            ])
            shared.CreateFaceVertexCountsAttr([3, 3, 3, 3])
            shared.CreateFaceVertexIndicesAttr([
                0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3,
            ])
            UsdPhysics.CollisionAPI.Apply(shared.GetPrim())
            UsdGeom.Xform.Define(stage, "/World/Collisions")
            collider = UsdGeom.Cube.Define(stage, "/World/Collisions/Collider")
            collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            stage.GetRootLayer().Save()

            scene = runtime.make_usd_scene_spec(
                asset_path=str(asset), root_path="/", fixed_base=True,
                ground_enabled=False, ground_height=0.0, self_collisions=False,
                show_colliders=False, home_positions={}, rigid_bodies=[],
                convex_decomposition_patterns=[], friction_overrides={},
            )
            scene["workspace_edits"] = {"visibility": {}}
            viewer = _NODE_REGISTRY["NewtonViewerConfig"]({
                "port": _free_port(), "label": "USD collision guide test",
                "show_grid": False,
            })["viewer"]
            run_id = f"newton-usd-colliders-{uuid.uuid4().hex[:8]}"
            try:
                started = runtime.start_session(
                    run_id, scene, viewer, "cpu", 60, 2, 4,
                    1.0e6, 1.0e4, {}, 45.0, 2.0,
                )
                by_path = {item["path"]: item for item in started["render_shapes"]}
                self.assertTrue(by_path["/World/Visual"]["visual"])
                self.assertFalse(by_path["/World/Visual"]["collider"])
                self.assertTrue(by_path["/World/Shared"]["visual"])
                self.assertFalse(by_path["/World/Shared"]["collider"])
                shared_proxy_path = runtime._collision_display_path("/World/Shared")
                self.assertFalse(by_path[shared_proxy_path]["visual"])
                self.assertTrue(by_path[shared_proxy_path]["collider"])
                self.assertEqual(
                    by_path[shared_proxy_path]["source_path"], "/World/Shared"
                )
                self.assertNotIn("/World/Collisions/Collider", by_path)
                collider_proxy_path = runtime._collision_display_path(
                    "/World/Collisions/Collider"
                )
                self.assertFalse(by_path[collider_proxy_path]["visual"])
                self.assertTrue(by_path[collider_proxy_path]["collider"])
                session = runtime.get_session(run_id)
                live_viewer = session.viewer
                render_stage = Usd.Stage.Open(session.render_asset_path)
                self.assertNotEqual(
                    UsdGeom.Imageable(
                        render_stage.GetPrimAtPath("/World/Visual")
                    ).ComputeVisibility(),
                    UsdGeom.Tokens.invisible,
                )
                self.assertFalse(
                    render_stage.GetPrimAtPath(runtime.VISUAL_DISPLAY_ROOT).IsValid()
                )
                overlay_text = Path(session.render_asset_path).read_text(encoding="utf-8")
                self.assertNotIn("point3f[] points", overlay_text)
                session.set_render_options(False, True)
                self.assertIs(session.viewer, live_viewer)
                self.assertFalse(session.viewer._viewer.show_visual)
                self.assertTrue(session.viewer._viewer.show_collision)
            finally:
                runtime.control_session(run_id, "stop")

    def test_generated_ground_exposes_live_ovrtx_transform_and_material_state(self) -> None:
        import copy
        import types

        from blacknode.pkg.blacknode_newton import runtime

        original_provider = runtime._WORKSPACE_VIEWER_PROVIDER
        original_scene = copy.deepcopy(runtime._WORKSPACE_SCENE_SPEC)
        original_state = copy.deepcopy(runtime._WORKSPACE_EDITOR_STATE)
        original_status = runtime.session_status
        original_get_session = runtime.get_session
        transform = {
            "translate_m": [0.5, -0.25, 0.1],
            "rotate_deg": [0.0, 0.0, 10.0],
            "scale": [1.5, 1.5, 1.0],
        }
        material = {
            "base_color": [0.2, 0.4, 0.9],
            "metallic": 0.7,
            "roughness": 0.2,
            "opacity": 0.95,
        }
        try:
            runtime._WORKSPACE_VIEWER_PROVIDER = "ovrtx"
            runtime._WORKSPACE_SCENE_SPEC = runtime.make_empty_scene_spec()
            runtime._WORKSPACE_EDITOR_STATE = runtime._default_workspace_editor_state()
            runtime._WORKSPACE_EDITOR_STATE["transforms"][runtime.WORKSPACE_GROUND_PATH] = transform
            runtime._WORKSPACE_EDITOR_STATE["materials"][runtime.WORKSPACE_GROUND_PATH] = material
            runtime.session_status = lambda _run_id: {
                "running": True,
                "viewer_provider": "ovrtx",
                "scene_items": [],
            }
            runtime.get_session = lambda _run_id: types.SimpleNamespace(
                viewer=types.SimpleNamespace(
                    selected_path="/BlacknodeOVRT/Ground/Geometry", _up_axis="z"
                )
            )

            status = runtime._workspace_status()
            ground = next(
                item for item in status["scene_items"]
                if item["path"] == runtime.WORKSPACE_GROUND_PATH
            )
            self.assertTrue(ground["editable"])
            self.assertTrue(ground["material_editable"])
            self.assertEqual(ground["material_path"], runtime.WORKSPACE_GROUND_MATERIAL_PATH)
            self.assertEqual(ground["transform"], transform)
            self.assertEqual(ground["material"], material)
            lights = next(
                item for item in status["scene_items"]
                if item["path"] == runtime.WORKSPACE_LIGHTS_PATH
            )
            key_light = next(
                item for item in status["scene_items"]
                if item["path"] == runtime.WORKSPACE_KEY_LIGHT_PATH
            )
            hdri_light = next(
                item for item in status["scene_items"]
                if item["path"] == runtime.WORKSPACE_HDRI_LIGHT_PATH
            )
            self.assertTrue(lights["visible"])
            self.assertEqual(key_light["type_name"], "DistantLight")
            self.assertTrue(key_light["visible"])
            self.assertEqual(key_light["light"]["intensity"], 2500.0)
            self.assertEqual(key_light["light"]["rotation_deg"], [-35.0, 25.0, -25.0])
            self.assertEqual(hdri_light["type_name"], "DomeLight")
            self.assertTrue(hdri_light["visible"])
            self.assertEqual(hdri_light["light"]["intensity"], 1.0)
            self.assertEqual(status["selected_path"], runtime.WORKSPACE_GROUND_PATH)
            self.assertEqual(
                runtime._require_workspace_item(
                    "/BlacknodeOVRT/Ground/Geometry", editable=True
                )["path"],
                runtime.WORKSPACE_GROUND_PATH,
            )
            selected = runtime.control_workspace(
                "select", {"path": "/BlacknodeOVRT/Ground/Geometry"}
            )
            self.assertEqual(selected["selected_path"], runtime.WORKSPACE_GROUND_PATH)
            environment_updates = []
            runtime.get_session = lambda _run_id: types.SimpleNamespace(
                viewer=types.SimpleNamespace(selected_path="", _up_axis="z"),
                set_environment=lambda environment: environment_updates.append(environment),
            )
            hidden_lights = runtime.control_workspace(
                "set_visibility",
                {"path": runtime.WORKSPACE_LIGHTS_PATH, "visible": False},
            )
            self.assertFalse(hidden_lights["environment"]["hdri_enabled"])
            self.assertFalse(
                hidden_lights["environment"]["distant_light"]["enabled"]
            )
            edited_light = runtime.control_workspace("set_light", {
                "path": runtime.WORKSPACE_KEY_LIGHT_PATH,
                "enabled": True,
                "intensity": 1800,
                "color": "#3366cc",
                "angle_deg": 6,
                "rotation_deg": [-25, 10, 35],
            })
            self.assertEqual(
                edited_light["environment"]["distant_light"]["intensity"], 1800.0
            )
            self.assertEqual(len(environment_updates), 2)
        finally:
            runtime._WORKSPACE_VIEWER_PROVIDER = original_provider
            runtime._WORKSPACE_SCENE_SPEC = original_scene
            runtime._WORKSPACE_EDITOR_STATE = original_state
            runtime.session_status = original_status
            runtime.get_session = original_get_session

    def test_ovstage_visibility_write_uses_a_live_token_attribute(self) -> None:
        if importlib.util.find_spec("ovstage") is None:
            self.skipTest("optional OVStage package is not installed")
        worker_path = PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "worker.py"
        spec = importlib.util.spec_from_file_location("blacknode_newton_ovrtx_visibility_test", worker_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Operation:
            def wait(self) -> None:
                return None

        class Paths:
            def __init__(self) -> None:
                self.created: list[list[str]] = []

            def create_path_list_from_strings(self, paths):
                self.created.append(list(paths))
                return "path-list"

            def intern_token(self, value: str) -> int:
                return {"visibility": 11, "inherited": 12, "invisible": 13}[value]

            def destroy_path_list(self, value) -> None:
                self.destroyed = value

        class Stage:
            def __init__(self) -> None:
                self.write = None

            def query_from_path_list(self, value):
                return "query"

            def write_attribute(self, query, attribute, **kwargs):
                self.write = (query, attribute, kwargs)
                return Operation()

            def release_query(self, query):
                return Operation()

        stage = Stage()
        paths = Paths()
        module._write_visibility(stage, paths, "/World/Robot", False, 7)
        self.assertEqual(paths.created, [["/World/Robot"]])
        self.assertEqual(stage.write[0:2], ("query", 11))
        self.assertEqual(stage.write[2]["ordinal"], 7)
        self.assertEqual(stage.write[2]["tensors"].tolist(), [13])

        import ovstage

        class MaterialPaths(Paths):
            def intern_token(self, value: str) -> int:
                return abs(hash(value))

            def intern_path(self, value: str) -> int:
                return 99

        class MaterialStage(Stage):
            def __init__(self) -> None:
                self.writes: list[tuple[object, object, dict[str, object]]] = []

            def write_attribute(self, query, attribute, **kwargs):
                self.writes.append((query, attribute, kwargs))
                return Operation()

        material_stage = MaterialStage()
        material_paths = MaterialPaths()
        module._write_stage_update(
            material_stage,
            material_paths,
            {
                "type": "material",
                "path": "/World/Robot",
                "material_path": "/__BlacknodeMaterials/Robot",
                "material": {
                    "base_color": [0.1, 0.6, 0.2],
                    "metallic": 0.3,
                    "roughness": 0.4,
                    "opacity": 1.0,
                },
            },
            8,
        )
        self.assertEqual(
            material_stage.writes[0][2]["semantic"],
            ovstage.AttributeSemantic.COLOR,
        )
        self.assertEqual(
            material_stage.writes[-1][2]["semantic"],
            ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        )

    def test_ovrtx_camera_frames_source_bounds_in_authored_units_and_up_axis(self) -> None:
        provider_path = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
        )
        spec = importlib.util.spec_from_file_location("blacknode_newton_ovrtx_provider_test", provider_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "centimetre_scene.usda"
            asset.write_text(
                '''#usda 1.0\n(\n    upAxis = "Y"\n    metersPerUnit = 0.01\n)\n\ndef Cube "Box"\n{\n    double size = 100\n    double3 xformOp:translate = (100, 50, -25)\n    uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n''',
                encoding="utf-8",
            )
            info = module._source_stage_info(str(asset))

        self.assertAlmostEqual(info["meters_per_unit"], 0.01)
        self.assertEqual(info["up_axis"], "y")
        session = type("Session", (), {"state_0": type("State", (), {"body_q": None})()})()
        position, target, up_axis = module._scene_camera(session, {"camera": {}}, info)
        self.assertEqual(up_axis, "y")
        self.assertEqual(target, [100.0, 50.0, -25.0])
        self.assertGreater(position[1], target[1])

    def test_ovrtx_accepts_an_empty_newton_stage(self) -> None:
        if importlib.util.find_spec("ovrtx") is None or importlib.util.find_spec("ovstage") is None:
            self.skipTest("optional OVRT release train is not installed")
        package = load_package(
            PACKAGE_ROOT,
            component_overrides={"viewer-ovrtx": True, "rosbridge": False},
        )
        self.assertTrue(package.ok, package.error)
        from blacknode.pkg.blacknode_newton.viewer_ovrtx.provider import _state_transforms

        state = type("EmptyNewtonState", (), {"body_q": None})()
        self.assertEqual(_state_transforms(state), [])

        import numpy as np
        import ovstage

        dtype = ovstage.numpy_to_dldatatype(np.dtype("float64"), lanes=16)
        self.assertEqual(dtype.bits, 64)
        self.assertEqual(dtype.lanes, 16)

    def test_generic_scene_contract_resolves_usd_and_validates_optional_bodies(self) -> None:
        result = _NODE_REGISTRY["NewtonUSDScene"]({
            "root_path": "/so101_new_calib",
            "rigid_bodies": [{
                "name": "test_box", "shape": "box", "position_m": [0.2455508, 0.0, 0.015],
                "size_m": [0.025, 0.025, 0.025], "mass_kg": 0.03,
            }],
        })
        self.assertTrue(result["ok"], result["report"])
        scene = result["scene"]
        self.assertEqual(scene["kind"], "blacknode.newton-scene")
        self.assertEqual(scene["schema_version"], 2)
        self.assertEqual(
            scene["source_asset"], "package://blacknode-newton/assets/so101_robot.usd"
        )
        self.assertTrue(Path(scene["asset_path"]).is_file())
        self.assertFalse(scene["self_collisions"])
        self.assertEqual(scene["render"], {"show_colliders": False})
        self.assertEqual(scene["rigid_bodies"][0]["position_m"][0], 0.2455508)
        self.assertEqual(scene["rigid_bodies"][0]["mass_kg"], 0.03)

    def test_model_free_scene_node_dispatches_robot_formats_and_xacro_arguments(self) -> None:
        missing = _NODE_REGISTRY["NewtonScene"]({"asset_path": ""})
        self.assertFalse(missing["ok"])
        self.assertIn("asset_path is required", missing["report"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usd_path = root / "viewer.usda"
            usd_path.write_text("#usda 1.0\n", encoding="utf-8")
            usd = _NODE_REGISTRY["NewtonScene"]({"asset_path": str(usd_path)})
            self.assertTrue(usd["ok"], usd["report"])
            self.assertEqual(usd["scene"]["root_path"], "/")
            self.assertIn("Newton USDA scene ready", usd["report"])

            urdf_path = root / "viewer.urdf"
            urdf_path.write_text(
                '<robot name="viewer"><link name="base_link"/></robot>',
                encoding="utf-8",
            )
            urdf = _NODE_REGISTRY["NewtonScene"]({"asset_path": str(urdf_path)})
            self.assertTrue(urdf["ok"], urdf["report"])
            self.assertEqual(urdf["scene"]["asset_format"], "urdf")
            self.assertFalse(urdf["scene"].get("robot_description_xml"))

            xacro_path = root / "viewer.urdf.xacro"
            xacro_path.write_text(
                """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="viewer">
  <xacro:arg name="link_name" default="default_link"/>
  <link name="$(arg link_name)"/>
</robot>
""",
                encoding="utf-8",
            )
            xacro = _NODE_REGISTRY["NewtonScene"]({
                "asset_path": str(xacro_path),
                "xacro_arguments": '{"link_name":"configured_link"}',
            })
            self.assertTrue(xacro["ok"], xacro["report"])
            self.assertIn('link name="configured_link"', xacro["scene"]["robot_description_xml"])

            invalid_arguments = _NODE_REGISTRY["NewtonScene"]({
                "asset_path": str(xacro_path),
                "xacro_arguments": "[]",
            })
            self.assertFalse(invalid_arguments["ok"])
            self.assertIn("one JSON object", invalid_arguments["report"])

            mjcf_path = root / "viewer.mjcf"
            mjcf_path.write_text(
                '<mujoco model="viewer"><worldbody/></mujoco>',
                encoding="utf-8",
            )
            mjcf = _NODE_REGISTRY["NewtonScene"]({"asset_path": str(mjcf_path)})
            self.assertTrue(mjcf["ok"], mjcf["report"])
            self.assertEqual(mjcf["scene"]["asset_format"], "mjcf")

    def test_xacro_package_find_is_safe_inside_expressions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "src" / "viewer_description"
            package.mkdir(parents=True)
            (package / "package.xml").write_text(
                "<package><name>viewer_description</name></package>",
                encoding="utf-8",
            )
            (package / "config.yaml").write_text(
                "link_name: configured_link\nnote: робот\n",
                encoding="utf-8",
            )
            xacro_path = package / "viewer.urdf.xacro"
            xacro_path.write_text(
                """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="viewer">
  <xacro:property name="config" value="${xacro.load_yaml('$(find viewer_description)/config.yaml')}"/>
  <link name="${config['link_name']}"/>
</robot>
""",
                encoding="utf-8",
            )
            result = _NODE_REGISTRY["NewtonScene"]({"asset_path": str(xacro_path)})
            self.assertTrue(result["ok"], result["report"])
            self.assertIn('link name="configured_link"', result["scene"]["robot_description_xml"])

            disconnected_path = package / "disconnected.urdf.xacro"
            disconnected_path.write_text(
                """<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="viewer">
  <link name="root_a"/>
  <link name="root_b"/>
</robot>
""",
                encoding="utf-8",
            )
            disconnected = _NODE_REGISTRY["NewtonScene"]({
                "asset_path": str(disconnected_path),
            })
            self.assertFalse(disconnected["ok"])
            self.assertIn("multiple root links: root_a, root_b", disconnected["report"])

    def test_robot_viewer_template_is_packageable_and_contains_no_model(self) -> None:
        path = PACKAGE_ROOT / "templates" / "robot-viewer.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        report = validate_workflow(workflow)
        self.assertTrue(report.ok, [issue.message for issue in report.errors])
        self.assertEqual(workflow["node_meta"]["scene"]["type"], "NewtonScene")
        self.assertEqual(workflow["node_meta"]["scene"]["params"]["asset_path"], "")
        self.assertEqual(
            workflow["metadata"]["required_components"],
            ["blacknode-newton/runtime", "blacknode-newton/viewer-viser"],
        )
        view = workflow["metadata"]["operator_view"]
        self.assertEqual(view["id"], "robot-viewer")
        viewer = view["sections"][0]["widgets"][0]
        self.assertEqual(viewer["type"], "viewer")
        model_field = view["sections"][2]["widgets"][0]["items"][0]
        self.assertEqual(model_field["input"], "file_path")
        self.assertIn(".xacro", model_field["extensions"])
        self.assertIn(".urdf", model_field["extensions"])
        serialized = json.dumps(workflow).lower()
        self.assertNotIn("so101_robot.usd", serialized)
        self.assertNotIn("openarmx", serialized)

    def test_particle_fill_is_procedural_bounded_and_overlap_checked(self) -> None:
        valid = _NODE_REGISTRY["NewtonUSDScene"]({
            "particle_fill": {
                "name": "test_grain",
                "position_m": [0.3, 0.1, 0.01],
                "dimensions": [10, 12, 8],
                "spacing_m": 0.0021,
                "radius_m": 0.001,
                "mass_kg": 3.2e-6,
                "jitter_m": 0.00005,
                "friction": 0.55,
                "color_rgb": [0.82, 0.58, 0.18],
                "container_collision_proxy": {
                    "body_path": "/World/Bin",
                    "source_shape_path": "/World/Bin/Collision",
                    "friction": 0.5,
                    "boxes": [
                        {
                            "position_m": [0.0, 0.0, -0.01],
                            "size_m": [0.05, 0.05, 0.002],
                        }
                    ],
                },
            },
        })
        self.assertTrue(valid["ok"], valid["report"])
        fill = valid["scene"]["particle_fill"]
        self.assertEqual(fill["particle_count"], 960)
        self.assertEqual(fill["spacing_m"], [0.0021, 0.0021, 0.0021])
        self.assertEqual(fill["color_rgb"], [0.82, 0.58, 0.18])
        self.assertEqual(
            fill["container_collision_proxy"]["source_shape_path"],
            "/World/Bin/Collision",
        )

        overlap = _NODE_REGISTRY["NewtonUSDScene"]({
            "particle_fill": {
                "position_m": [0, 0, 0], "dimensions": [2, 2, 2],
                "spacing_m": 0.002, "radius_m": 0.001, "jitter_m": 0.0001,
            },
        })
        self.assertFalse(overlap["ok"])
        self.assertIn("spacing minus jitter", overlap["report"])

        excessive = _NODE_REGISTRY["NewtonUSDScene"]({
            "particle_fill": {
                "position_m": [0, 0, 0], "dimensions": [100, 100, 11],
                "spacing_m": 0.0021, "radius_m": 0.001,
            },
        })
        self.assertFalse(excessive["ok"])
        self.assertIn("cannot exceed 100000", excessive["report"])

    def test_default_workspace_scene_is_portable_and_has_no_generated_ground(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        scene = runtime.make_default_workspace_scene_spec()
        asset = Path(scene["asset_path"])
        self.assertEqual(
            asset,
            PACKAGE_ROOT / "assets" / "scenes" / "so101_tabletop.usd",
        )
        self.assertTrue(asset.is_file())
        self.assertEqual(scene["root_path"], "/so101_new_calib")
        self.assertFalse(scene["ground"]["enabled"])
        friction = scene["collision"]["friction_overrides"]
        self.assertEqual(friction["/env/green_cube"], 3.0)
        self.assertEqual(
            scene["collision"]["mass_overrides"],
            runtime.DEFAULT_PICK_MASS_OVERRIDES,
        )
        self.assertEqual(
            scene["collision"]["static_body_patterns"],
            list(runtime.DEFAULT_STATIC_COLLISION_BODY_PATTERNS),
        )
        self.assertEqual(
            scene["collision"]["mesh_approximation_overrides"],
            runtime.DEFAULT_MESH_APPROXIMATION_OVERRIDES,
        )
        self.assertEqual(
            scene["collision"]["convex_decomposition_patterns"],
            list(runtime.DEFAULT_PICK_CONVEX_DECOMPOSITION_PATTERNS),
        )
        self.assertEqual(
            scene["collision"]["grip_pads"],
            list(runtime.DEFAULT_GRIP_PAD_SPECS),
        )
        self.assertEqual(
            scene["collision"]["compound_shape_proxies"],
            list(runtime.DEFAULT_CONTAINER_COLLISION_PROXIES),
        )
        self.assertNotIn("grasp_assist", scene["collision"])
        self.assertFalse(runtime.make_empty_scene_spec()["ground"]["enabled"])
        self.assertFalse(runtime._default_workspace_editor_state()["show_grid"])
        self.assertEqual(
            runtime._default_workspace_editor_state()["joint_drive_overrides"]["gripper"],
            runtime.DEFAULT_GRIPPER_DRIVE,
        )
        self.assertEqual(
            runtime._default_workspace_editor_state()["environment"]["hdri"],
            "apartment",
        )
        self.assertEqual(
            runtime._default_workspace_editor_state()["environment"]["intensity"],
            1.0,
        )

    def test_urdf_and_xacro_robot_descriptions_use_the_native_newton_path(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package.xml").write_text(
                '''<package format="3">
  <name>test_description</name><version>0.0.0</version>
  <description>test</description><maintainer email="test@example.com">Test</maintainer>
  <license>MIT</license>
</package>''',
                encoding="utf-8",
            )
            meshes = root / "meshes"
            meshes.mkdir()
            (meshes / "link.stl").write_text(
                '''solid link
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 0.1 0 0
vertex 0 0.1 0
endloop
endfacet
endsolid link
''',
                encoding="utf-8",
            )
            urdf = root / "two_link.urdf"
            urdf.write_text(
                '''<robot name="two_link">
  <link name="base">
    <visual><geometry><box size="0.4 0.3 0.2"/></geometry></visual>
    <collision><geometry><box size="0.4 0.3 0.2"/></geometry></collision>
  </link>
  <link name="arm">
    <visual><geometry><cylinder radius="0.05" length="0.5"/></geometry></visual>
    <collision><geometry><cylinder radius="0.05" length="0.5"/></geometry></collision>
  </link>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.35"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
</robot>''',
                encoding="utf-8",
            )
            xacro_path = root / "two_link.xacro"
            xacro_path.write_text(
                '''<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="macro_robot">
  <xacro:property name="side" value="0.25"/>
  <link name="mesh_link">
    <visual><geometry><mesh filename="file://$(find test_description)/meshes/link.stl"/></geometry></visual>
    <collision><geometry><mesh filename="package://test_description/meshes/link.stl"/></geometry></collision>
  </link>
  <joint name="fixed" type="fixed"><parent link="mount"/><child link="mesh_link"/></joint>
</robot>''',
                encoding="utf-8",
            )
            environment_name = "BLACKNODE_XACRO_MACHINE_TYPE_TEST"
            environment_xacro_path = root / "environment_robot.xacro"
            environment_xacro_path.write_text(
                f'''<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="environment_robot">
  <xacro:property name="machine" value="$(env {environment_name})"/>
  <link name="${{machine}}"/>
</robot>''',
                encoding="utf-8",
            )

            urdf_scene = runtime.make_robot_description_scene_spec(
                asset_path=str(urdf), fixed_base=True, ground_enabled=True,
                ground_height=0.0, self_collisions=False, show_colliders=False,
            )
            xacro_scene = runtime.make_robot_description_scene_spec(
                asset_path=str(xacro_path), fixed_base=True, ground_enabled=True,
                ground_height=0.0, self_collisions=False, show_colliders=False,
            )
            self.assertEqual(urdf_scene["asset_format"], "urdf")
            self.assertEqual(urdf_scene["robot_description_format"], "urdf")
            self.assertFalse(urdf_scene["robot_description_xml"])
            self.assertEqual(xacro_scene["robot_description_format"], "xacro")
            self.assertIn('<link name="mount"', xacro_scene["robot_description_xml"])
            self.assertIn((root / "meshes" / "link.stl").as_posix(), xacro_scene["robot_description_xml"])
            self.assertNotIn("package://test_description", xacro_scene["robot_description_xml"])

            previous_environment = os.environ.pop(environment_name, None)
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    f"Xacro requires environment variable '{environment_name}'",
                ):
                    runtime.make_robot_description_scene_spec(
                        asset_path=str(environment_xacro_path), fixed_base=True,
                        ground_enabled=True, ground_height=0.0,
                        self_collisions=False, show_colliders=False,
                    )
                environment_scene = runtime.make_robot_description_scene_spec(
                    asset_path=str(environment_xacro_path), fixed_base=True,
                    ground_enabled=True, ground_height=0.0,
                    self_collisions=False, show_colliders=False,
                    xacro_environment={environment_name: "ROSOrin_Mecanum"},
                )
                self.assertIn('<link name="ROSOrin_Mecanum"', environment_scene["robot_description_xml"])
                self.assertNotIn(environment_name, os.environ)
                with self.assertRaisesRegex(ValueError, "invalid Xacro environment variable name"):
                    runtime.make_robot_description_scene_spec(
                        asset_path=str(environment_xacro_path), fixed_base=True,
                        ground_enabled=True, ground_height=0.0,
                        self_collisions=False, show_colliders=False,
                        xacro_environment={"INVALID-NAME": "value"},
                    )
            finally:
                if previous_environment is not None:
                    os.environ[environment_name] = previous_environment

            run_id = f"newton-urdf-{uuid.uuid4().hex[:8]}"
            viewer = _NODE_REGISTRY["NewtonViewerConfig"]({
                "port": _free_port(), "label": "URDF contract test", "show_grid": True,
            })["viewer"]
            try:
                started = runtime.start_session(
                    run_id, urdf_scene, viewer, "cpu", 60, 2, 4,
                    1.0e6, 1.0e4, {}, 45.0, 2.0,
                )
                self.assertTrue(started["running"], started)
                self.assertEqual(started["joint_names"], ["arm_joint"])
                session = runtime.get_session(run_id)
                self.assertIsNotNone(session)
                self.assertTrue(any(shape["visual"] for shape in started["render_shapes"]))
                self.assertTrue(any(shape["collider"] for shape in started["render_shapes"]))
                live_viewer = session.viewer
                render_status = session.set_render_options(False, True)
                self.assertIs(session.viewer, live_viewer)
                self.assertFalse(render_status["show_visuals"])
                self.assertTrue(render_status["show_colliders"])
                self.assertFalse(session.viewer._viewer.show_visual)
                self.assertTrue(session.viewer._viewer.show_collision)
                self.assertTrue(Path(session.render_asset_path).is_file())
                self.assertTrue(any("two_link/" in item["name"] for item in session.scene_items))
            finally:
                runtime.control_session(run_id, "stop")

            xacro_run_id = f"newton-xacro-{uuid.uuid4().hex[:8]}"
            xacro_viewer = _NODE_REGISTRY["NewtonViewerConfig"]({
                "port": _free_port(), "label": "Xacro contract test", "show_grid": True,
            })["viewer"]
            try:
                xacro_started = runtime.start_session(
                    xacro_run_id, xacro_scene, xacro_viewer, "cpu", 60, 2, 4,
                    1.0e6, 1.0e4, {}, 45.0, 2.0,
                )
                self.assertTrue(xacro_started["running"], xacro_started)
                self.assertTrue(xacro_started["render_shapes"])
            finally:
                runtime.control_session(xacro_run_id, "stop")

    def test_mjcf_include_scene_uses_the_native_newton_path(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = root / "robot.xml"
            robot.write_text(
                '''<mujoco model="included robot">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <joint damping="0.2"/>
    <default class="visual"><geom contype="0" conaffinity="0"/></default>
    <default class="collision"><geom/></default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0.4">
      <freejoint/>
      <geom class="visual" type="box" size="0.15 0.1 0.08" rgba="0.2 0.4 0.8 1"/>
      <geom class="visual" type="box" size="0.15 0.1 0.08" pos="0.04 0 0" rgba="0.2 0.4 0.8 1"/>
      <geom class="collision" type="box" size="0.15 0.1 0.08"/>
      <body name="arm" pos="0 0 0.1">
        <joint name="arm_joint" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom class="visual" type="capsule" size="0.04 0.18" pos="0 0 0.18"/>
        <geom class="collision" type="capsule" size="0.04 0.18" pos="0 0 0.18"/>
      </body>
    </body>
  </worldbody>
  <keyframe><key name="home" qpos="0 0 0.35 1 0 0 0 0.4"/></keyframe>
</mujoco>''',
                encoding="utf-8",
            )
            scene_path = root / "scene.xml"
            scene_path.write_text(
                '''<mujoco model="include scene">
  <include file="robot.xml"/>
  <worldbody><geom name="floor" type="plane" size="0 0 0.05"/></worldbody>
</mujoco>''',
                encoding="utf-8",
            )
            invalid = root / "not_mujoco.xml"
            invalid.write_text("<robot/>", encoding="utf-8")

            scene = runtime.make_mjcf_scene_spec(
                asset_path=str(scene_path), fixed_base=None, ground_enabled=None,
                ground_height=0.0, self_collisions=False, show_colliders=False,
            )
            robot_only = runtime.make_mjcf_scene_spec(
                asset_path=str(robot), fixed_base=None, ground_enabled=None,
                ground_height=0.0, self_collisions=False, show_colliders=False,
            )
            self.assertEqual(scene["asset_format"], "mjcf")
            self.assertIsNone(scene["fixed_base"])
            self.assertFalse(scene["ground"]["enabled"])
            self.assertEqual(scene["mjcf_home_qpos"], [
                0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.4,
            ])
            self.assertTrue(robot_only["ground"]["enabled"])
            with self.assertRaisesRegex(ValueError, "not a MuJoCo model"):
                runtime.make_mjcf_scene_spec(
                    asset_path=str(invalid), fixed_base=None, ground_enabled=None,
                    ground_height=0.0, self_collisions=False, show_colliders=False,
                )

            run_id = f"newton-mjcf-{uuid.uuid4().hex[:8]}"
            viewer = _NODE_REGISTRY["NewtonViewerConfig"]({
                "port": _free_port(), "label": "MJCF contract test", "show_grid": False,
            })["viewer"]
            try:
                started = runtime.start_session(
                    run_id, scene, viewer, "cpu", 60, 2, 4,
                    1.0e6, 1.0e4, {}, 45.0, 2.0,
                )
                self.assertTrue(started["running"], started)
                self.assertEqual(started["joint_names"], ["arm_joint"])
                session = runtime.get_session(run_id)
                self.assertIsNotNone(session)
                self.assertNotEqual(
                    session.joint_indices["arm_joint"],
                    session.joint_drive_indices["arm_joint"],
                )
                self.assertAlmostEqual(session.home["arm_joint"], 0.4)
                model_q = session.model.joint_q.numpy().tolist()
                self.assertEqual(model_q[3:7], [0.0, 0.0, 0.0, 1.0])
                self.assertTrue(any(shape["visual"] for shape in started["render_shapes"]))
                self.assertTrue(any(shape["collider"] for shape in started["render_shapes"]))
                self.assertEqual(
                    len({shape["path"] for shape in started["render_shapes"]}),
                    len(started["render_shapes"]),
                )
                self.assertTrue(any(
                    "/instance_1" in shape["path"]
                    for shape in started["render_shapes"]
                ))
                self.assertTrue(Path(session.render_asset_path).is_file())
                from pxr import Usd, UsdGeom

                render_stage = Usd.Stage.Open(session.render_asset_path)
                self.assertIsNotNone(render_stage)
                visual_path = next(
                    shape["path"]
                    for shape in started["render_shapes"]
                    if shape["visual"] and not shape["collider"]
                )
                collider_path = next(
                    shape["path"]
                    for shape in started["render_shapes"]
                    if shape["collider"] and not shape["visual"]
                )
                self.assertEqual(
                    UsdGeom.Imageable(render_stage.GetPrimAtPath(visual_path))
                    .GetVisibilityAttr().Get(Usd.TimeCode(0)),
                    UsdGeom.Tokens.inherited,
                )
                self.assertEqual(
                    UsdGeom.Imageable(render_stage.GetPrimAtPath(collider_path))
                    .GetVisibilityAttr().Get(Usd.TimeCode(0)),
                    UsdGeom.Tokens.invisible,
                )
                for shape in started["render_shapes"]:
                    authored_visibility = UsdGeom.Imageable(
                        render_stage.GetPrimAtPath(shape["path"])
                    ).GetVisibilityAttr().Get(Usd.TimeCode(0))
                    self.assertEqual(
                        authored_visibility != UsdGeom.Tokens.invisible,
                        bool(shape["visual"]),
                    )
                provider_path = (
                    PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
                )
                provider_spec = importlib.util.spec_from_file_location(
                    "blacknode_newton_mjcf_ovrtx_provider_test", provider_path
                )
                self.assertIsNotNone(provider_spec)
                self.assertIsNotNone(provider_spec.loader)
                provider_module = importlib.util.module_from_spec(provider_spec)
                provider_spec.loader.exec_module(provider_module)
                worker_path = (
                    PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "worker.py"
                )
                worker_spec = importlib.util.spec_from_file_location(
                    "blacknode_newton_mjcf_ovrtx_worker_test", worker_path
                )
                self.assertIsNotNone(worker_spec)
                self.assertIsNotNone(worker_spec.loader)
                worker_module = importlib.util.module_from_spec(worker_spec)
                worker_spec.loader.exec_module(worker_module)
                body_entries = provider_module._body_entries(
                    session, session.model, session.render_asset_path
                )
                self.assertTrue(body_entries)
                self.assertTrue(all(
                    entry["path"] != collider_path for entry in body_entries
                ))
                transforms = session.state_0.body_q.numpy().tolist()
                for entry in body_entries:
                    authored_local = UsdGeom.Xformable(
                        render_stage.GetPrimAtPath(entry["path"])
                    ).GetLocalTransformation(Usd.TimeCode(0.0))
                    streamed_local = worker_module._body_pose_matrix(
                        entry, transforms, 1.0
                    )
                    self.assertTrue(all(
                        math.isclose(
                            float(authored_local[row][column]),
                            float(streamed_local[row * 4 + column]),
                            abs_tol=1.0e-5,
                        )
                        for row in range(4)
                        for column in range(4)
                    ))
            finally:
                runtime.control_session(run_id, "stop")

    def test_workflow_template_is_portable_and_valid(self) -> None:
        workflow = json.loads(
            (PACKAGE_ROOT / "templates" / "usd-scene-viewer.json").read_text(encoding="utf-8")
        )
        report = validate_workflow(workflow)
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(workflow["kind"], "blacknode.workflow")
        self.assertEqual(workflow["schema_version"], 1)
        self.assertEqual(workflow["metadata"]["required_packages"], ["blacknode-newton"])
        self.assertEqual(workflow["entrypoint"], {"node_id": "viewer_url", "port": "value"})

        grain_workflow = json.loads(
            (PACKAGE_ROOT / "templates" / "so101-grain-spill-demo.json").read_text(
                encoding="utf-8"
            )
        )
        grain_report = validate_workflow(grain_workflow)
        self.assertTrue(grain_report.ok, grain_report.to_dict())
        grain_scene = grain_workflow["node_meta"]["scene"]["params"]
        self.assertEqual(grain_scene["particle_fill"]["dimensions"], [18, 18, 10])
        self.assertEqual(math.prod(grain_scene["particle_fill"]["dimensions"]), 3240)
        self.assertFalse(grain_scene["ground_enabled"])
        self.assertEqual(grain_scene["convex_decomposition_patterns"], [])
        self.assertEqual(
            len(
                grain_scene["particle_fill"]["container_collision_proxy"]["boxes"]
            ),
            5,
        )
        self.assertEqual(
            grain_workflow["node_meta"]["viewer"]["params"]["show_grid"], False
        )
        self.assertEqual(
            grain_workflow["node_meta"]["viewer"]["params"]["hdri"], "apartment"
        )
        self.assertEqual(
            grain_workflow["node_meta"]["viewer"]["params"]["hdri_intensity"], 1.0
        )
        self.assertEqual(
            grain_workflow["node_meta"]["viewer"]["params"]["render_fps"], 20
        )
        self.assertEqual(
            grain_workflow["node_meta"]["simulation"]["params"]["fps"], 24
        )
        self.assertEqual(
            grain_workflow["node_meta"]["simulation"]["params"]["substeps"], 8
        )
        self.assertEqual(
            grain_workflow["node_meta"]["simulation"]["params"]["solver_iterations"], 4
        )
        grain_graph = graph_from_workflow(grain_workflow)
        cooked_scene = grain_graph._cook("scene", "scene")
        self.assertTrue(
            Path(cooked_scene["asset_path"]).as_posix().endswith(
                "assets/scenes/so101_tabletop.usd"
            )
        )
        self.assertEqual(cooked_scene["particle_fill"]["particle_count"], 3240)
        self.assertFalse(cooked_scene["ground"]["enabled"])
        cooked_viewer = grain_graph._cook("viewer", "viewer")
        self.assertEqual(cooked_viewer["provider"], "viser")
        self.assertFalse(cooked_viewer["show_grid"])
        self.assertEqual(
            grain_workflow["metadata"]["required_components"],
            ["blacknode-newton/runtime", "blacknode-newton/viewer-viser"],
        )

        sync_workflow = json.loads(
            (PACKAGE_ROOT / "templates" / "ros2-newton-joint-sync.json").read_text(encoding="utf-8")
        )
        sync_report = validate_workflow(sync_workflow)
        self.assertTrue(sync_report.ok, sync_report.to_dict())
        self.assertEqual(
            sync_workflow["metadata"]["required_components"],
            [
                "blacknode-newton/runtime",
                "blacknode-newton/rosbridge",
                "blacknode-ros2/rosbridge",
            ],
        )
        self.assertEqual(sync_workflow["node_meta"]["ros_bridge"]["params"]["direction"], "ros_to_newton")
        self.assertEqual(sync_workflow["node_meta"]["ros_bridge"]["params"]["action"], "start")

        replay_workflow = json.loads(
            (PACKAGE_ROOT / "templates" / "dataset-episode-newton-replay.json").read_text(
                encoding="utf-8"
            )
        )
        replay_report = validate_workflow(replay_workflow)
        self.assertTrue(replay_report.ok, replay_report.to_dict())
        self.assertEqual(
            replay_workflow["metadata"]["required_components"],
            [
                "blacknode-newton/runtime",
                "blacknode-newton/replay",
                "blacknode-dataset/recording",
                "blacknode-dataset/publishing",
            ],
        )
        self.assertEqual(
            replay_workflow["node_meta"]["arm_workspace"]["params"]["run_id"],
            "__blacknode_newton_workspace__",
        )
        self.assertEqual(replay_workflow["node_meta"]["resume_workspace"]["params"]["action"], "resume")
        self.assertEqual(replay_workflow["node_meta"]["replay_bridge"]["params"]["action"], "start")

    def test_unloaded_viewer_provider_fails_with_structured_detail(self) -> None:
        from blacknode.pkg.blacknode_newton.viewer_contract import create_viewer

        with self.assertRaisesRegex(RuntimeError, "loaded providers: .*viser"):
            create_viewer("ovui", None, None, {})

    def test_viser_avoids_a_loopback_port_owned_by_another_application(self) -> None:
        from blacknode.pkg.blacknode_newton.viewer_viser.provider import (
            _select_viewer_port,
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            requested = int(occupied.getsockname()[1])
            selected = _select_viewer_port(requested)
        self.assertNotEqual(selected, requested)
        self.assertGreaterEqual(selected, 1024)

    def test_viewer_config_normalizes_portable_view_settings(self) -> None:
        result = _NODE_REGISTRY["NewtonViewerConfig"]({
            "background_color": "#A0b1C2",
            "show_grid": False,
            "hdri": "studio",
            "show_hdri_background": False,
            "hdri_intensity": 1.75,
            "render_fps": 24,
            "camera_position": [1, 2, 3],
            "camera_target": [0, 0, 0.5],
            "camera_up_axis": "Z",
            "camera_speed": 3,
        })
        self.assertTrue(result["ok"], result["report"])
        self.assertEqual(result["viewer"]["background_color"], "#a0b1c2")
        self.assertFalse(result["viewer"]["show_grid"])
        self.assertEqual(
            result["viewer"]["environment"],
            {"hdri": "studio", "show_background": False, "intensity": 1.75},
        )
        self.assertEqual(result["viewer"]["camera"]["position_m"], [1.0, 2.0, 3.0])
        self.assertEqual(result["viewer"]["camera"]["up_axis"], "z")
        self.assertEqual(result["viewer"]["camera"]["speed_m_s"], 3.0)
        self.assertEqual(result["viewer"]["render_fps"], 24)

        invalid = _NODE_REGISTRY["NewtonViewerConfig"]({
            "camera_position": [1, 2, 3], "camera_target": [],
        })
        self.assertFalse(invalid["ok"])
        self.assertIn("must both be empty", invalid["report"])

    def test_workspace_environment_switches_cleanly_between_custom_and_presets(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.exr"
            custom.write_bytes(b"placeholder")
            selected = runtime._updated_workspace_environment(
                runtime._default_workspace_editor_state()["environment"],
                {"hdri": "custom", "hdri_path": str(custom), "intensity": 2.5},
            )
            self.assertEqual(selected["hdri"], "custom")
            self.assertEqual(selected["hdri_path"], str(custom.resolve()))
            self.assertEqual(selected["intensity"], 2.5)
            preset = runtime._updated_workspace_environment(selected, {"hdri": "forest"})
            self.assertEqual(preset["hdri"], "forest")
            self.assertEqual(preset["hdri_path"], "")
            hidden = runtime._updated_workspace_environment(
                preset, {
                    "show_background": False,
                    "hdri_enabled": False,
                    "intensity": 0.0,
                    "distant_light": {
                        "enabled": False,
                        "intensity": 1250,
                        "color": "#3366cc",
                        "angle_deg": 8,
                        "rotation_deg": [-20, 15, 40],
                    },
                }
            )
            self.assertFalse(hidden["show_background"])
            self.assertFalse(hidden["hdri_enabled"])
            self.assertEqual(hidden["intensity"], 0.0)
            self.assertEqual(hidden["distant_light"], {
                "enabled": False,
                "intensity": 1250.0,
                "color": "#3366cc",
                "angle_deg": 8.0,
                "rotation_deg": [-20.0, 15.0, 40.0],
            })
            with self.assertRaisesRegex(ValueError, "unknown HDRI preset"):
                runtime._updated_workspace_environment(hidden, {"hdri": "missing"})

    def test_joint_drive_settings_are_validated(self) -> None:
        from blacknode.pkg.blacknode_newton.runtime import NewtonSession

        self.assertEqual(NewtonSession._validate_drive_gain("gain", 0), 0.0)
        self.assertEqual(
            NewtonSession._validate_drive_overrides({"gripper": {"damping": 50000}}),
            {"gripper": {"damping": 50000.0}},
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1e12"):
            NewtonSession._validate_drive_gain("gain", -1)

    def test_mujoco_gravity_compensation_selects_robot_not_free_props(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        actgravcomp = SimpleNamespace(values=None)
        body_gravcomp = SimpleNamespace(values=None)
        builder = SimpleNamespace(
            custom_attributes={
                "mujoco:jnt_actgravcomp": actgravcomp,
                "mujoco:gravcomp": body_gravcomp,
            },
            # Bodies 0-2 form the robot; body 3 is a separate free prop.
            joint_parent=[-1, 0, 1, -1],
            joint_child=[0, 1, 2, 3],
            body_mass=[0.0, 1.0, 0.25, 0.005],
        )
        result = runtime._configure_mujoco_gravity_compensation(
            builder,
            {"shoulder": 0, "gripper": 1},
            {"shoulder": 1, "gripper": 2},
            {
                "enabled": True,
                "joint_names": ["shoulder"],
                "body_factor": 1.0,
            },
        )

        self.assertEqual(actgravcomp.values, {0: True})
        self.assertEqual(body_gravcomp.values, {1: 1.0, 2: 1.0})
        self.assertNotIn(3, body_gravcomp.values)
        self.assertEqual(result["joint_names"], ["shoulder"])
        self.assertEqual(result["body_indices"], [1, 2])

        actgravcomp.values = {}
        body_gravcomp.values = {}
        selected_result = runtime._configure_mujoco_gravity_compensation(
            builder,
            {"shoulder": 0, "gripper": 1},
            {"shoulder": 1, "gripper": 2},
            {
                "joint_names": ["shoulder"],
                "body_mode": "selected",
            },
        )
        self.assertEqual(body_gravcomp.values, {1: 1.0})
        self.assertEqual(selected_result["body_indices"], [1])
        self.assertEqual(selected_result["body_mode"], "selected")

        with self.assertRaisesRegex(ValueError, "unknown one-DOF joints"):
            runtime._configure_mujoco_gravity_compensation(
                builder,
                {"shoulder": 0},
                {"shoulder": 1},
                {"joint_names": ["missing"]},
            )

    def test_digital_twin_history_is_sampled_bounded_and_clearable(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        session = runtime.NewtonSession(
            "trace-test",
            {"kind": "blacknode.newton-scene"},
            {"kind": "blacknode.newton-viewer"},
            "cpu", 60, 4, 16, 1.0e8, 1.0e5, {}, 45.0, 2.0,
        )
        session.joint_indices = {"shoulder": 0}
        session.current = {"shoulder": 0.25}
        session.status = lambda: {"ok": True}
        with (
            mock.patch.object(runtime, "DIGITAL_TWIN_HISTORY_LIMIT", 3),
            mock.patch.object(
                runtime.time,
                "time",
                side_effect=[100.0, 100.01, 100.1, 100.2, 100.3],
            ),
        ):
            for reference in (0.1, 0.2, 0.3, 0.4, 0.5):
                session.record_joint_observation(
                    {"shoulder": reference}, source="replay:test", stale_after_seconds=0.5
                )
        self.assertEqual(len(session.digital_twin_history), 3)
        self.assertEqual(
            [sample["received_at"] for sample in session.digital_twin_history],
            [100.1, 100.2, 100.3],
        )
        self.assertAlmostEqual(
            session.digital_twin_history[-1]["joint_errors"]["shoulder"], -0.25
        )
        self.assertEqual(session.clear_digital_twin_history(), {"ok": True})
        self.assertEqual(session.digital_twin_history, [])

        session.joint_units = {"shoulder": "radians"}
        with mock.patch.object(runtime.time, "time", return_value=101.0):
            session.record_joint_observation(
                {"shoulder": 0.1}, source="replay:test", stale_after_seconds=0.5
            )
        self.assertEqual(
            session.set_digital_twin_ghost(True, "overlay"), {"ok": True}
        )
        self.assertEqual(session.digital_twin_ghost["offset_m"], [0.0, 0.0, 0.0])
        with mock.patch.object(session, "_update_reference_state") as update_reference:
            session.digital_twin_ghost["visible"] = False
            with mock.patch.object(runtime.time, "time", return_value=101.01):
                session.record_joint_observation(
                    {"shoulder": 0.1}, source="replay:test", stale_after_seconds=0.5
                )
            update_reference.assert_not_called()
            session.set_digital_twin_ghost(True, "overlay")
            update_reference.assert_called_once_with()
        session.set_digital_twin_ghost(False, "custom", [0.5, -0.25, 0.1])
        self.assertEqual(session.digital_twin_ghost["offset_m"], [0.5, -0.25, 0.1])
        with self.assertRaisesRegex(ValueError, "overlay, beside, or custom"):
            session.set_digital_twin_ghost(True, "unknown")
        with mock.patch.object(runtime.time, "time", return_value=101.1):
            with self.assertRaisesRegex(RuntimeError, "disarmed"):
                session.sync_simulation_to_external_pose()
            session.armed = True
            with mock.patch.object(
                session, "command", return_value={"synced": True}
            ) as command:
                self.assertEqual(
                    session.sync_simulation_to_external_pose(), {"synced": True}
                )
            command.assert_called_once_with(
                {"shoulder": 0.1}, source="digital-twin-sync-once"
            )
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.dict(os.environ, {"BLACKNODE_CONFIG_DIR": directory}),
                mock.patch.object(runtime, "get_session", return_value=session),
            ):
                artifact = runtime.save_digital_twin_artifact(
                    "trace-test", "Pick trace", asset_path="robot.usd", scene_label="Robot"
                )
                artifact_path = Path(artifact["path"])
                self.assertTrue(artifact_path.is_file())
                self.assertEqual(artifact_path.parent, Path(directory).resolve() / "newton-runs")
                self.assertEqual(artifact["kind"], "blacknode.newton-run-artifact")
                self.assertEqual(artifact["summary"]["sample_count"], 1)
                listed = runtime.list_digital_twin_artifacts()
                self.assertEqual([item["artifact_id"] for item in listed], [artifact["artifact_id"]])
                loaded = runtime.load_digital_twin_baseline("trace-test", artifact["artifact_id"])
                self.assertEqual(loaded, {"ok": True})
                self.assertEqual(session.digital_twin_baseline["name"], "Pick trace")
                self.assertEqual(session.digital_twin_baseline["matched_joint_names"], ["shoulder"])
                runtime.clear_digital_twin_baseline("trace-test")
                self.assertEqual(session.digital_twin_baseline, {})
                with self.assertRaisesRegex(ValueError, "invalid Newton run-artifact id"):
                    runtime.load_digital_twin_baseline("trace-test", "../outside")

    def test_robot_monitor_pose_maps_calibrated_degrees_into_newton_units(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        session = mock.Mock()
        session.joint_indices = {"shoulder_pan": 0, "elbow_flex": 1}
        session.joint_units = {"shoulder_pan": "radians", "elbow_flex": "radians"}

        mapped = runtime._mapped_external_joint_positions(
            session,
            {"shoulder_pan": 90.0, "elbow_flex": -45.0, "camera_tilt": 12.0},
            "degree",
        )

        self.assertAlmostEqual(mapped["shoulder_pan"], math.pi / 2.0)
        self.assertAlmostEqual(mapped["elbow_flex"], -math.pi / 4.0)
        remapped = runtime._mapped_external_joint_positions(
            session,
            {"leader_shoulder": 180.0, "ignored": 1.0},
            "degrees",
            {"leader_shoulder": "shoulder_pan"},
        )
        self.assertAlmostEqual(remapped["shoulder_pan"], math.pi)
        with self.assertRaisesRegex(ValueError, "unsupported external joint position unit"):
            runtime._mapped_external_joint_positions(session, {"shoulder_pan": 20}, "ticks")
        with self.assertRaisesRegex(ValueError, "no joints matching"):
            runtime._mapped_external_joint_positions(session, {"unknown": 20}, "degree")

    def test_robot_monitor_workspace_ingest_records_and_safely_follows(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        session = mock.Mock()
        session.joint_indices = {"shoulder_pan": 0}
        session.joint_units = {"shoulder_pan": "radians"}
        session.home = {"shoulder_pan": 0.25}
        session.record_joint_observation = mock.Mock()
        session.follow_joint_observation = mock.Mock()
        session.record_and_follow_joint_observation = mock.Mock()
        session.start_stream_follow = mock.Mock()
        session.stop_stream_follow = mock.Mock()
        session.set_digital_twin_ghost = mock.Mock()
        session.set_paused = mock.Mock()
        session.set_armed = mock.Mock()
        session.command = mock.Mock()
        stream_status = {
            "open": True,
            "armed": True,
            "simulation_running": True,
            "phase": "running",
            "accepted": True,
            "command_count": 2,
            "clamped": [],
        }
        session.record_and_follow_joint_observation.return_value = stream_status
        session.stream_control_status = mock.Mock(return_value=stream_status)
        with (
            mock.patch.object(runtime, "get_session", return_value=session),
            mock.patch.object(runtime, "_workspace_status", return_value={"open": True}),
        ):
            runtime.control_workspace("start_robot_monitor_follow", {
                "source": "robot-monitor:usb-1",
                "stale_after_seconds": 0.5,
            })
            session.set_digital_twin_ghost.assert_called_once_with(
                False, "overlay", compact=True
            )
            session.start_stream_follow.assert_called_once_with("robot-monitor:usb-1", 0.5)
            result = runtime.control_workspace("robot_monitor_sample", {
                "source": "robot-monitor:usb-1",
                "positions": {"shoulder_pan": 90.0, "extra": 1.0},
                "position_unit": "degree",
                "home_positions": {"shoulder_pan": 30.0},
                "home_position_unit": "degree",
                "observed_at": 100.0,
                "age_seconds": 0.05,
                "stale_after_seconds": 0.5,
                "available": True,
                "stale": False,
                "connected": True,
                "calibrated": True,
                "follow": True,
            })
            self.assertEqual(result, stream_status)
            session.record_and_follow_joint_observation.assert_called_once()
            recorded = session.record_and_follow_joint_observation.call_args.args[0]
            self.assertAlmostEqual(recorded["shoulder_pan"], math.pi / 6.0 + math.pi / 2.0)
            self.assertEqual(
                session.record_and_follow_joint_observation.call_args.kwargs,
                {
                    "source": "robot-monitor:usb-1",
                    "observed_at": 100.0,
                    "stale_after_seconds": 0.5,
                },
            )
            session.record_and_follow_joint_observation.reset_mock()
            runtime.control_workspace("robot_monitor_sample", {
                "source": "robot-monitor:usb-1",
                "positions": {"shoulder_pan": 0.0},
                "position_unit": "degree",
                "home_positions": {"shoulder_pan": 30.0},
                "home_position_unit": "degree",
                "age_seconds": 0.0,
                "stale_after_seconds": 0.5,
                "available": True,
                "stale": False,
                "connected": True,
                "calibrated": True,
                "follow": True,
            })
            latest = session.record_and_follow_joint_observation.call_args.args[0]
            self.assertAlmostEqual(latest["shoulder_pan"], math.pi / 6.0)
            session.record_and_follow_joint_observation.assert_called_once_with(
                latest,
                source="robot-monitor:usb-1",
                observed_at=None,
                stale_after_seconds=0.5,
            )
            with self.assertRaisesRegex(RuntimeError, "telemetry is stale"):
                runtime.control_workspace("robot_monitor_sample", {
                    "positions": {"shoulder_pan": 90.0},
                    "position_unit": "degree",
                    "available": True,
                    "stale": True,
                    "connected": True,
                    "calibrated": True,
                })
            with self.assertRaisesRegex(RuntimeError, "active calibration"):
                runtime.control_workspace("robot_monitor_sample", {
                    "positions": {"shoulder_pan": 90.0},
                    "position_unit": "degree",
                    "available": True,
                    "stale": False,
                    "connected": True,
                    "calibrated": False,
                })
            runtime.control_workspace(
                "stop_robot_monitor_follow", {"source": "robot-monitor:usb-1"}
            )
            session.stop_stream_follow.assert_called_once_with("robot-monitor:usb-1")
            session.stop_stream_follow.reset_mock()
            session.set_digital_twin_ghost.reset_mock()
            home_result = runtime.control_workspace("robot_monitor_home", {
                "source": "robot-monitor:usb-1:calibration-home",
                "positions": {"shoulder_pan": 30.0},
                "position_unit": "degree",
                "calibrated": True,
            })
            self.assertEqual(home_result, {"open": True})
            session.stop_stream_follow.assert_called_once_with("")
            session.set_digital_twin_ghost.assert_called_once_with(False, "overlay")
            session.set_paused.assert_called_once_with(False)
            session.set_armed.assert_called_once_with(True)
            commanded = session.command.call_args.args[0]
            self.assertAlmostEqual(commanded["shoulder_pan"], math.pi / 6.0)
            self.assertEqual(
                session.command.call_args.kwargs["source"],
                "robot-monitor:usb-1:calibration-home",
            )
            with self.assertRaisesRegex(RuntimeError, "active calibration"):
                runtime.control_workspace("robot_monitor_home", {
                    "positions": {"shoulder_pan": 30.0},
                    "position_unit": "degree",
                    "calibrated": False,
                })

        real_session = runtime.NewtonSession(
            "robot-monitor-watchdog",
            {"kind": "blacknode.newton-scene"},
            {"kind": "blacknode.newton-viewer"},
            "cpu", 60, 4, 16, 1.0e8, 1.0e5, {}, 45.0, 2.0,
        )
        real_session.joint_indices = {"shoulder_pan": 0}
        real_session.joint_limits = {"shoulder_pan": (-math.pi, math.pi)}
        real_session.current = {"shoulder_pan": 0.0}
        real_session.desired = {"shoulder_pan": 0.0}
        real_session.applied = {"shoulder_pan": 0.0}
        real_session.phase = "paused"
        real_session.paused = True
        real_session.status = lambda: {"armed": real_session.armed}
        real_session.start_stream_follow("robot-monitor:usb-1", 0.5)
        self.assertTrue(real_session.armed)
        self.assertFalse(real_session.paused)
        real_session.set_paused(True)
        real_session.follow_joint_observation(
            {"shoulder_pan": 0.25}, "robot-monitor:usb-1", 0.5
        )
        self.assertFalse(real_session.paused)
        self.assertEqual(real_session.phase, "running")
        real_session.stream_follow_deadline = time.monotonic() + 0.01
        record_started = threading.Event()
        expiry_results: list[bool] = []
        original_record = real_session.record_joint_observation

        def delayed_record(*args, **kwargs):
            record_started.set()
            time.sleep(0.05)
            return original_record(*args, **kwargs)

        def expire_while_recording() -> None:
            record_started.wait(1.0)
            time.sleep(0.02)
            expiry_results.append(real_session._expire_stale_stream_follow())

        expiry_thread = threading.Thread(target=expire_while_recording)
        with mock.patch.object(
            real_session, "record_joint_observation", side_effect=delayed_record
        ):
            expiry_thread.start()
            atomic = real_session.record_and_follow_joint_observation(
                {"shoulder_pan": 0.5}, "robot-monitor:usb-1", stale_after_seconds=0.5
            )
            expiry_thread.join(1.0)
        self.assertEqual(expiry_results, [False])
        self.assertTrue(atomic["armed"])
        compact = real_session.stream_control_status()
        self.assertEqual(
            set(compact),
            {
                "open", "armed", "simulation_running", "phase", "accepted",
                "command_count", "clamped",
            },
        )
        self.assertTrue(real_session.stream_follow_deadline)
        self.assertTrue(
            real_session._expire_stale_stream_follow(real_session.stream_follow_deadline + 0.01)
        )
        self.assertFalse(real_session.armed)
        self.assertEqual(real_session.desired, real_session.applied)
        recovered = real_session.follow_joint_observation(
            {"shoulder_pan": 0.75}, "robot-monitor:usb-1", 0.5
        )
        self.assertTrue(recovered["armed"])
        self.assertEqual(real_session.desired["shoulder_pan"], 0.75)
        real_session.set_armed(False)
        real_session.set_armed(True)
        home_target = dict(real_session.desired)
        real_session.follow_joint_observation(
            {"shoulder_pan": 1.0}, "robot-monitor:usb-1", 0.5
        )
        self.assertEqual(real_session.desired, home_target)


class BlacknodeNewtonLivePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package = load_package(
            PACKAGE_ROOT,
            component_overrides={"viewer-ovrtx": False, "rosbridge": False},
        )
        if not package.ok:
            raise AssertionError(package.error)

    def test_bundled_so101_runs_on_newton_and_serves_browser_teleoperation(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        run_id = f"newton-test-{uuid.uuid4().hex[:8]}"
        port = _free_port()
        scene_result = _NODE_REGISTRY["NewtonUSDScene"]({
            "root_path": "/so101_new_calib",
            "home_positions": {
                "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                "wrist_flex": 1.57, "wrist_roll": 0.0, "gripper": 0.5,
            },
            "rigid_bodies": [{
                "name": "test_box", "shape": "box", "position_m": [0.2455508, 0.0, 0.015],
                "size_m": [0.025, 0.025, 0.025], "mass_kg": 0.03,
                "friction": 0.8, "color_rgb": [0.92, 0.18, 0.12],
            }],
            "convex_decomposition_patterns": ["gripper_link", "moving_jaw_"],
            "friction_overrides": {"gripper_link": 3.0, "moving_jaw_": 3.0},
        })
        viewer_result = _NODE_REGISTRY["NewtonViewerConfig"](
            {
                "port": port, "label": "Blacknode Newton integration test",
                "background_color": "#203040", "show_grid": False,
                "hdri": "studio", "show_hdri_background": True, "hdri_intensity": 1.25,
                "camera_position": [1.1, 0.8, 1.0], "camera_target": [0.0, 0.0, 0.2],
                "camera_up_axis": "Z", "camera_speed": 2.5,
            }
        )
        self.assertTrue(scene_result["ok"], scene_result["report"])
        self.assertTrue(viewer_result["ok"], viewer_result["report"])

        try:
            started = runtime.start_session(
                run_id,
                scene_result["scene"],
                viewer_result["viewer"],
                "auto",
                60,
                4,
                16,
                1.0e8,
                1.0e5,
                {"gripper": {"stiffness": 5.0e7, "damping": 5.0e4}},
                45.0,
                2.0,
            )
            self.assertTrue(started["running"], started)
            self.assertFalse(started["armed"])
            self.assertGreaterEqual(started["active_collision_shapes"], 17)
            self.assertEqual(started["joint_names"][-1], "gripper")
            self.assertEqual(started["joint_units"]["gripper"], "radians")
            self.assertEqual(
                started["joint_drive_gains"]["gripper"],
                {"stiffness": 5.0e7, "damping": 5.0e4},
            )
            self.assertEqual(started["viewer_port"], port)
            self.assertEqual(started["viewer_requested_port"], port)
            self.assertIn("viser", started["available_viewers"])
            session = runtime.get_session(run_id)
            self.assertIsNotNone(session)
            self.assertIsNotNone(session.viewer._viewer._camera_request)
            self.assertEqual(session.viewer._background_rgb, (32, 48, 64))
            self.assertFalse(session.viewer._grid_visible)
            self.assertEqual(session.viewer._hdri, "studio")
            self.assertEqual(session.viewer._hdri_intensity, 1.25)
            self.assertEqual(session.viewer._viewer.camera_speed, 2.5)
            self.assertEqual(session.viewer._camera_view[0], [1.1, 0.8, 1.0])
            live_viewer = session.viewer
            live_status = session.set_joint_properties("gripper", 4.0e7, 4.0e4)
            self.assertIs(session.viewer, live_viewer)
            self.assertEqual(
                live_status["joint_drive_gains"]["gripper"],
                {"stiffness": 4.0e7, "damping": 4.0e4},
            )
            dynamics = live_status["joint_dynamics"]["gripper"]
            self.assertTrue(dynamics["child_body"])
            self.assertGreater(dynamics["child_body_mass_kg"], 0.0)
            self.assertEqual(len(dynamics["child_body_inertia_kg_m2"]), 3)
            motion_status = session.set_joint_motion_limits(
                "gripper", math.radians(90.0), math.radians(5.0)
            )
            self.assertIs(session.viewer, live_viewer)
            self.assertAlmostEqual(
                motion_status["joint_motion_limits"]["gripper"]["max_velocity"],
                math.radians(90.0),
            )
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                session.set_joint_motion_limits("gripper", math.radians(721.0), 0.1)
            self.assertNotIn("/blacknode/ground_reference", session.viewer._viewer._scene_handles)
            _wait_for(
                lambda: len(session.viewer._viewer._scene_handles) > 0,
                5.0,
                "Viser did not publish render handles for the USD scene",
            )
            self.assertEqual(session.viewer._viewer._plane_handles, {})

            with urllib.request.urlopen(started["viewer_url"], timeout=5.0) as response:
                body = response.read(512)
                self.assertEqual(response.status, 200)
                self.assertIn(b"<!doctype html>", body.lower())
                self.assertIsNone(response.headers.get("X-Frame-Options"))

            with self.assertRaisesRegex(RuntimeError, "disarmed"):
                runtime.command_session(run_id, {"shoulder_lift": 0.3})

            settled = _wait_for(
                lambda: runtime.session_status(run_id)
                if runtime.session_status(run_id)["frame_count"] >= 30
                else None,
                15.0,
                "Newton did not advance 30 physics frames",
            )
            self.assertEqual(settled["phase"], "running", settled)
            self.assertFalse(settled["last_error"], settled)
            self.assertAlmostEqual(
                settled["rigid_body_positions_m"]["test_box"][2], 0.0125, delta=0.01
            )

            reference = settled["positions"]["shoulder_lift"] + 0.05
            observed = runtime.record_joint_observation(
                run_id,
                {"shoulder_lift": reference},
                source="rosbridge:integration-test",
                observed_at=time.time(),
                stale_after_seconds=0.5,
            )
            twin = observed["digital_twin"]
            self.assertTrue(twin["available"])
            self.assertFalse(twin["stale"])
            self.assertEqual(twin["source"], "rosbridge:integration-test")
            self.assertEqual(twin["matched_joint_count"], 1)
            self.assertAlmostEqual(twin["reference_positions"]["shoulder_lift"], reference)
            self.assertAlmostEqual(
                twin["joint_errors"]["shoulder_lift"],
                observed["positions"]["shoulder_lift"] - reference,
            )
            self.assertAlmostEqual(
                twin["max_abs_error"], abs(twin["joint_errors"]["shoulder_lift"])
            )
            self.assertTrue(twin["ghost"]["visible"])
            self.assertEqual(twin["ghost"]["placement"], "beside")
            self.assertTrue(session.articulation_body_indices)
            ovrtx_provider_path = (
                PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
            )
            ovrtx_spec = importlib.util.spec_from_file_location(
                "blacknode_newton_ovrtx_ghost_entries_test", ovrtx_provider_path
            )
            self.assertIsNotNone(ovrtx_spec)
            self.assertIsNotNone(ovrtx_spec.loader)
            ovrtx_provider = importlib.util.module_from_spec(ovrtx_spec)
            ovrtx_spec.loader.exec_module(ovrtx_provider)
            tracked_visuals = [
                entry
                for entry in session.render_shapes
                if entry.get("visual") and int(entry.get("body_index", -1)) >= 0
            ]
            self.assertGreaterEqual(len(tracked_visuals), 17)
            self.assertTrue(all(
                len(entry.get("body_bind_world_matrix") or []) == 16
                for entry in tracked_visuals
            ))
            collision_wireframes = ovrtx_provider._collision_wireframes(
                session.render_asset_path, session.render_shapes
            )
            self.assertGreaterEqual(len(collision_wireframes), 17)
            self.assertTrue(all(frame["edges"] for frame in collision_wireframes))
            ovrtx_body_entries = ovrtx_provider._body_entries(
                session, session.model, session.render_asset_path
            )
            self.assertFalse(any(
                str(entry.get("path") or "").startswith("/so101_new_calib/")
                for entry in ovrtx_body_entries
            ))
            ghost_entries = ovrtx_provider._ghost_entries(
                session, session.model, session.render_asset_path
            )
            self.assertTrue(ghost_entries)
            self.assertTrue(all(
                entry["index"] in session.articulation_body_indices
                for entry in ghost_entries
            ))
            ghost_status = session.set_digital_twin_ghost(True, "overlay")
            self.assertIs(session.viewer, live_viewer)
            self.assertEqual(ghost_status["digital_twin"]["ghost"]["offset_m"], [0.0, 0.0, 0.0])
            _wait_for(
                lambda: session.viewer._viewer._layers["real_reference"].visible,
                5.0,
                "Viser did not show the live real-pose ghost layer",
            )

            before = settled["positions"]["shoulder_lift"]
            runtime.control_session(run_id, "arm")
            accepted = runtime.command_session(
                run_id, {"shoulder_lift": 0.3}, source="integration-test"
            )
            self.assertEqual(accepted["command_count"], 1)
            self.assertEqual(accepted["last_command_source"], "integration-test")
            moved = _wait_for(
                lambda: runtime.session_status(run_id)
                if abs(
                    runtime.session_status(run_id)["positions"]["shoulder_lift"]
                    - before
                )
                > 0.03
                else None,
                10.0,
                "USD articulation did not respond to an armed joint target",
            )
            self.assertTrue(moved["running"], moved)
            self.assertFalse(moved["last_error"], moved)
        finally:
            stopped = runtime.control_session(run_id, "stop")
            self.assertFalse(stopped["running"], stopped)

    def test_particle_fill_advances_and_viser_publishes_the_grain_batch(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        scene = runtime.make_usd_scene_spec(
            asset_path=runtime.package_asset_uri("assets/so101_robot.usd"),
            root_path="/so101_new_calib",
            fixed_base=True,
            ground_enabled=True,
            ground_height=0.0,
            self_collisions=False,
            show_colliders=False,
            home_positions={},
            rigid_bodies=[],
            convex_decomposition_patterns=[],
            friction_overrides={},
            particle_fill={
                "name": "test_grain",
                "position_m": [0.4, 0.2, 0.05],
                "dimensions": [4, 4, 4],
                "spacing_m": 0.0105,
                "radius_m": 0.005,
                "mass_kg": 0.001,
                "jitter_m": 0.0002,
                "friction": 0.55,
            },
        )
        viewer = _NODE_REGISTRY["NewtonViewerConfig"]({
            "port": _free_port(), "label": "Particle contract test", "show_grid": False,
        })["viewer"]
        run_id = f"newton-particles-{uuid.uuid4().hex[:8]}"
        try:
            started = runtime.start_session(
                run_id, scene, viewer, "auto", 60, 4, 8,
                1.0e8, 1.0e5, {}, 45.0, 2.0,
            )
            self.assertEqual(started["particle_count"], 64)
            advanced = _wait_for(
                lambda: runtime.session_status(run_id)
                if runtime.session_status(run_id)["frame_count"] >= 3
                else None,
                10.0,
                "Newton particle fill did not advance",
            )
            self.assertFalse(advanced["last_error"], advanced)
            session = runtime.get_session(run_id)
            self.assertTrue(session.viewer._viewer.show_particles)
            particle_key = _wait_for(
                lambda: next(
                    (
                        key for key in session.viewer._viewer._scene_handles
                        if "particle" in key.lower()
                    ),
                    None,
                ),
                5.0,
                "Viser did not publish the particle render batch",
            )
            particle_handle = session.viewer._viewer._scene_handles[particle_key]
            frame_before = runtime.session_status(run_id)["frame_count"]
            _wait_for(
                lambda: runtime.session_status(run_id)["frame_count"] >= frame_before + 3,
                5.0,
                "Newton particle renderer did not advance",
            )
            self.assertIs(
                session.viewer._viewer._scene_handles[particle_key], particle_handle
            )
            self.assertEqual(particle_handle.point_shading, "gradient")
            self.assertEqual(particle_handle.precision, "float32")
            positions = session.state_0.particle_q.numpy().tolist()
            self.assertEqual(len(positions), 64)
            self.assertTrue(all(math.isfinite(float(value)) for row in positions for value in row))
        finally:
            stopped = runtime.control_session(run_id, "stop")
            self.assertFalse(stopped["running"], stopped)

    def test_default_newton_workspace_opens_tabletop_paused_without_ground_or_grid(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        try:
            opened = runtime.control_workspace("open")
            self.assertTrue(opened["open"], opened)
            self.assertFalse(opened["simulation_running"], opened)
            self.assertEqual(opened["phase"], "paused")
            self.assertEqual(opened["scene_label"], "so101_tabletop.usd")
            self.assertFalse(opened["show_grid"])
            self.assertTrue(opened["viewer_url"].startswith("http://127.0.0.1:"))
            workspace = runtime.get_session(runtime.WORKSPACE_RUN_ID)
            self.assertIsNotNone(workspace)
            self.assertFalse(workspace.scene["ground"]["enabled"])
            body_labels = [str(label) for label in workspace.model.body_label]
            body_masses = workspace.model.body_mass.numpy().tolist()
            for color in ("green", "blue", "red"):
                cube_suffix = f"/env/{color}_cube"
                cube_index = next(
                    index for index, label in enumerate(body_labels)
                    if label.endswith(cube_suffix)
                )
                self.assertAlmostEqual(float(body_masses[cube_index]), 0.03, places=5)
                container_index = next(
                    index for index, label in enumerate(body_labels)
                    if label.endswith(f"/env/{color}_container")
                )
                self.assertAlmostEqual(float(body_masses[container_index]), 0.03, places=5)
                self.assertIn(container_index, workspace.dynamic_body_indices)

            shape_bodies = workspace.model.shape_body.numpy().tolist()
            shape_labels = [str(label) for label in workspace.model.shape_label]
            shape_flags = workspace.model.shape_flags.numpy().tolist()
            newton, wp = workspace._imports()
            collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
            for color in ("green", "blue", "red"):
                container_suffix = f"/env/{color}_container/{color}_container"
                container_shape_index = next(
                    index for index, label in enumerate(workspace.model.shape_label)
                    if str(label).endswith(container_suffix)
                )
                container_body_index = next(
                    index for index, label in enumerate(body_labels)
                    if label.endswith(f"/env/{color}_container")
                )
                self.assertEqual(
                    int(shape_bodies[container_shape_index]), container_body_index
                )
                proxy_indices = [
                    index for index, label in enumerate(shape_labels)
                    if label.startswith(
                        f"/so101_new_calib{container_suffix}_proxy_"
                    )
                ]
                self.assertEqual(len(proxy_indices), 5)
                self.assertEqual(
                    {int(shape_bodies[index]) for index in proxy_indices},
                    {container_body_index},
                )
                self.assertFalse(
                    int(shape_flags[container_shape_index]) & collide_bit
                )
                self.assertTrue(
                    all(
                        int(shape_flags[index]) & collide_bit
                        for index in proxy_indices
                    )
                )
            active_table_object_indices = [
                index
                for index, label in enumerate(shape_labels)
                if "/so101_new_calib/env/" in label
                and int(shape_flags[index]) & collide_bit
            ]
            self.assertTrue(active_table_object_indices)
            for body_index in workspace.dynamic_body_indices:
                if "/env/" not in body_labels[body_index]:
                    continue
                self.assertTrue(
                    any(
                        int(shape_bodies[index]) == body_index
                        for index in active_table_object_indices
                    ),
                    body_labels[body_index],
                )
            for pattern in runtime.DEFAULT_PICK_FRICTION_OVERRIDES:
                self.assertGreater(
                    len(workspace.friction_override_matches.get(pattern) or []),
                    0,
                    pattern,
                )
            self.assertEqual(len(workspace.grip_pad_shape_indices), 2)
            expected_pad_bodies = {
                next(
                    index for index, label in enumerate(body_labels)
                    if label == spec["body_path"]
                )
                for spec in runtime.DEFAULT_GRIP_PAD_SPECS
            }
            self.assertEqual(
                {
                    int(shape_bodies[index])
                    for index in workspace.grip_pad_shape_indices
                },
                expected_pad_bodies,
            )
            expected_source_indices = {
                index
                for index, label in enumerate(shape_labels)
                if any(
                    spec["source_shape_pattern"] in label
                    for spec in runtime.DEFAULT_GRIP_PAD_SPECS
                )
            }
            self.assertEqual(
                set(workspace.grip_pad_source_shape_indices),
                expected_source_indices,
            )
            shape_mu = workspace.model.shape_material_mu.numpy().tolist()
            shape_margin = workspace.model.shape_margin.numpy().tolist()
            self.assertTrue(expected_source_indices)
            self.assertTrue(
                all(
                    not (int(shape_flags[index]) & collide_bit)
                    for index in expected_source_indices
                )
            )
            for shape_index in workspace.grip_pad_shape_indices:
                self.assertAlmostEqual(float(shape_mu[shape_index]), 1.2, places=5)
                self.assertAlmostEqual(float(shape_margin[shape_index]), 0.0001, places=6)
                self.assertTrue(int(shape_flags[shape_index]) & collide_bit)
            contact_pairs = {
                tuple(sorted((int(pair[0]), int(pair[1]))))
                for pair in workspace.model.shape_contact_pairs.numpy().tolist()
            }
            self.assertIn(
                tuple(sorted(workspace.grip_pad_shape_indices)),
                contact_pairs,
            )
            self.assertEqual(
                workspace.joint_drive_gains["gripper"],
                runtime.DEFAULT_GRIPPER_DRIVE,
            )
            gripper_drive_index = workspace.joint_drive_indices["gripper"]
            self.assertAlmostEqual(
                float(workspace.model.joint_target_ke.numpy()[gripper_drive_index]),
                runtime.DEFAULT_GRIPPER_DRIVE["stiffness"],
                places=2,
            )
            ground = next(
                item for item in opened["scene_items"]
                if item["path"] == runtime.WORKSPACE_GROUND_PATH
            )
            self.assertFalse(ground["visible"])

            playing = runtime.control_workspace("play")
            self.assertTrue(playing["simulation_running"], playing)
            advanced = _wait_for(
                lambda: runtime.control_workspace("status")
                if runtime.control_workspace("status")["frame_count"] >= 2
                else None,
                5.0,
                "Default Newton workspace did not advance physics",
            )
            self.assertFalse(advanced["last_error"], advanced)

            green_container_index = next(
                index
                for index, label in enumerate(body_labels)
                if label.endswith("/env/green_container")
            )
            green_cube_index = next(
                index
                for index, label in enumerate(body_labels)
                if label.endswith("/env/green_cube")
            )
            with workspace.lock:
                for state in (workspace.state_0, workspace.state_1):
                    poses = state.body_q.numpy().tolist()
                    container_position = poses[green_container_index][:3]
                    poses[green_cube_index][:3] = [
                        float(container_position[0]),
                        float(container_position[1]),
                        float(container_position[2]) + 0.06,
                    ]
                    state.body_q.assign(
                        wp.array(
                            poses,
                            dtype=wp.transform,
                            device=workspace.model.device,
                        )
                    )
                    velocities = state.body_qd.numpy().tolist()
                    velocities[green_cube_index] = [
                        0.0 for _value in velocities[green_cube_index]
                    ]
                    state.body_qd.assign(
                        wp.array(
                            velocities,
                            dtype=wp.spatial_vector,
                            device=workspace.model.device,
                        )
                    )
            settled_in_container = _wait_for(
                lambda: workspace.state_0.body_q.numpy().tolist()
                if (
                    workspace.state_0.body_q.numpy().tolist()[green_cube_index][2]
                    - workspace.state_0.body_q.numpy().tolist()[green_container_index][2]
                ) < 0.02
                else None,
                3.0,
                "Cube did not settle into the compound container collider",
            )
            container_position = settled_in_container[green_container_index][:3]
            cube_position = settled_in_container[green_cube_index][:3]
            self.assertLess(
                math.hypot(
                    cube_position[0] - container_position[0],
                    cube_position[1] - container_position[1],
                ),
                0.01,
            )
            self.assertGreater(cube_position[2] - container_position[2], -0.005)
            self.assertFalse(workspace.status()["last_error"])

            workspace.set_armed(True)
            workspace.command({"gripper": 0.9})
            _wait_for(
                lambda: workspace.current["gripper"] > 0.8,
                3.0,
                "Real-contact test gripper did not open",
            )
            from pxr import Gf

            blue_cube_index = next(
                index
                for index, label in enumerate(body_labels)
                if label.endswith("/env/blue_cube")
            )
            with workspace.lock:
                setup_poses = workspace.state_0.body_q.numpy().tolist()
                setup_pad_centers = []
                setup_pad_transforms = []
                for body_index, local_pose in workspace.grip_pad_bindings:
                    local = runtime._newton_pose_matrix_m(local_pose, Gf)
                    body = runtime._newton_pose_matrix_m(
                        setup_poses[body_index], Gf
                    )
                    setup_pad_transforms.append(local * body)
                    setup_pad_centers.append(
                        Gf.Transform(local * body).GetTranslation()
                    )
                midpoint = (setup_pad_centers[0] + setup_pad_centers[1]) * 0.5
                # Exercise the rendered forward tip, not merely the center of
                # the generated pad. The former short pad left this region of
                # both visible jaws with no collision coverage.
                setup_tip_axis = (
                    setup_pad_transforms[0].Transform(Gf.Vec3d(1.0, 0.0, 0.0))
                    - setup_pad_transforms[0].Transform(Gf.Vec3d(0.0, 0.0, 0.0))
                ).GetNormalized()
                tip_offset_m = 0.012
                setup_cube_position = midpoint + setup_tip_axis * tip_offset_m
                for state in (workspace.state_0, workspace.state_1):
                    poses = state.body_q.numpy().tolist()
                    poses[blue_cube_index][:3] = [
                        float(value)
                        for value in setup_cube_position
                    ]
                    state.body_q.assign(
                        wp.array(
                            poses,
                            dtype=wp.transform,
                            device=workspace.model.device,
                        )
                    )
                    velocities = state.body_qd.numpy().tolist()
                    velocities[blue_cube_index] = [
                        0.0 for _value in velocities[blue_cube_index]
                    ]
                    state.body_qd.assign(
                        wp.array(
                            velocities,
                            dtype=wp.spatial_vector,
                            device=workspace.model.device,
                        )
                    )
            workspace.command({"gripper": math.radians(-10.0)})

            cube_shape_indices = [
                index
                for index, body_index in enumerate(shape_bodies)
                if int(body_index) == blue_cube_index
            ]
            self.assertTrue(cube_shape_indices)

            def bilateral_physical_contact():
                count = min(
                    int(workspace.contacts.rigid_contact_count.numpy()[0]),
                    int(workspace.contacts.rigid_contact_max),
                )
                shape0 = workspace.contacts.rigid_contact_shape0.numpy()[:count].tolist()
                shape1 = workspace.contacts.rigid_contact_shape1.numpy()[:count].tolist()
                touching_pads = set()
                for raw_shape0, raw_shape1 in zip(shape0, shape1):
                    pair = (int(raw_shape0), int(raw_shape1))
                    for pad_index, other_index in (pair, pair[::-1]):
                        if (
                            pad_index in workspace.grip_pad_shape_indices
                            and other_index in cube_shape_indices
                        ):
                            touching_pads.add(pad_index)
                if (
                    touching_pads == set(workspace.grip_pad_shape_indices)
                    and abs(
                        workspace.applied["gripper"] - math.radians(-10.0)
                    ) < 1.0e-5
                ):
                    return workspace.status()
                return None

            contact = _wait_for(
                bilateral_physical_contact,
                3.0,
                "Both physical fingertip colliders did not contact the cube",
            )
            self.assertAlmostEqual(
                contact["targets"]["gripper"], math.radians(-10.0), places=5
            )
            self.assertAlmostEqual(
                contact["applied"]["gripper"], math.radians(-10.0), places=5
            )
            _wait_for(
                lambda: workspace.status()
                if 0.25 < workspace.current["gripper"] < 0.45
                else None,
                3.0,
                "Physical contact did not oppose the fully closed gripper target",
            )
            workspace.command({"wrist_roll": 1.0})
            _wait_for(
                lambda: workspace.current["wrist_roll"] > 0.9,
                3.0,
                "Wrist did not rotate during the real-contact grasp",
            )
            with workspace.lock:
                held_poses = workspace.state_0.body_q.numpy().tolist()
                pad_centers = []
                for body_index, local_pose in workspace.grip_pad_bindings:
                    local = runtime._newton_pose_matrix_m(local_pose, Gf)
                    body = runtime._newton_pose_matrix_m(
                        held_poses[body_index], Gf
                    )
                    pad_centers.append(
                        Gf.Transform(local * body).GetTranslation()
                    )
                held_midpoint = (pad_centers[0] + pad_centers[1]) * 0.5
                held_cube_position = Gf.Vec3d(*held_poses[blue_cube_index][:3])
            self.assertGreater(
                float((pad_centers[1] - pad_centers[0]).GetLength()),
                0.021,
            )
            self.assertLess(
                float((pad_centers[1] - pad_centers[0]).GetLength()),
                0.035,
            )
            self.assertLess(
                float((held_cube_position - held_midpoint).GetLength()),
                0.026,
            )
            self.assertGreater(
                float((held_cube_position - setup_cube_position).GetLength()),
                0.02,
            )
            live_flags = workspace.model.shape_flags.numpy().tolist()
            self.assertTrue(
                any(
                    int(live_flags[index]) & collide_bit
                    for index in cube_shape_indices
                )
            )
            workspace.command({"gripper": 0.9})
            _wait_for(
                lambda: workspace.current["gripper"] > 0.8,
                3.0,
                "Physical gripper did not reopen after releasing the cube",
            )

            stopped = runtime.control_workspace("stop")
            self.assertTrue(stopped["open"], stopped)
            self.assertFalse(stopped["simulation_running"], stopped)
            self.assertEqual(stopped["phase"], "paused")
        finally:
            closed = runtime.control_workspace("close")
            self.assertFalse(closed["open"], closed)


if __name__ == "__main__":
    unittest.main()
