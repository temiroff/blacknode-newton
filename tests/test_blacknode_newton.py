from __future__ import annotations

import json
import importlib.util
import socket
import sys
import tempfile
import time
import unittest
import urllib.request
import uuid
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BLACKNODE_ROOT = PACKAGE_ROOT.parents[1]
CORE_PYTHON = BLACKNODE_ROOT / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from blacknode.node import _NODE_REGISTRY  # noqa: E402
from blacknode.packages import load_package  # noqa: E402
from blacknode.workflow import validate_workflow  # noqa: E402


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
            component_overrides={"viewer-ovrtx": False, "rosbridge": False},
        )

    def test_default_package_loads_real_runtime_and_browser_provider(self) -> None:
        self.assertTrue(self.package.ok, self.package.error)
        self.assertEqual(self.package.enabled_components, ["runtime", "viewer-viser"])
        self.assertEqual(
            set(self.package.node_types),
            {
                "NewtonJointCommand",
                "NewtonSimulation",
                "NewtonUSDScene",
                "NewtonViewerConfig",
            },
        )
        from blacknode.pkg.blacknode_newton.viewer_contract import available_viewers

        self.assertEqual(available_viewers(), ["viser"])
        self.assertIn("viewer-ovrtx", self.package.components)
        self.assertFalse(self.package.components["viewer-ovrtx"]["enabled"])

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

        payload = module._matrix_array([matrix])
        self.assertEqual(str(payload.dtype), "float64")
        self.assertEqual(payload.shape, (16,))
        self.assertEqual(module._encode_frame({}, 90), b"")

        provider_path = (
            PACKAGE_ROOT / "components" / "viewer-ovrtx" / "nodes" / "provider.py"
        )
        provider_source = provider_path.read_text(encoding="utf-8")
        self.assertIn('draggable="false"', provider_source)
        self.assertIn("requestAnimationFrame", provider_source)

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
                "background_color": "#111827",
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
        self.assertIn('interpolation = "constant"', wrapper)
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/Grid"))
        self.assertTrue(layer.GetPrimAtPath("/BlacknodeOVRT/GridAxes"))
        self.assertIn("ovstage.population.open_usd_from_string(", worker_path.read_text(encoding="utf-8"))

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

        invalid = _NODE_REGISTRY["NewtonViewerConfig"]({
            "camera_position": [1, 2, 3], "camera_target": [],
        })
        self.assertFalse(invalid["ok"])
        self.assertIn("must both be empty", invalid["report"])

    def test_joint_drive_settings_are_validated(self) -> None:
        from blacknode.pkg.blacknode_newton.runtime import NewtonSession

        self.assertEqual(NewtonSession._validate_drive_gain("gain", 0), 0.0)
        self.assertEqual(
            NewtonSession._validate_drive_overrides({"gripper": {"damping": 50000}}),
            {"gripper": {"damping": 50000.0}},
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1e12"):
            NewtonSession._validate_drive_gain("gain", -1)


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

    def test_empty_newton_workspace_opens_paused_and_keeps_viewer_while_stopped(self) -> None:
        from blacknode.pkg.blacknode_newton import runtime

        try:
            opened = runtime.control_workspace("new")
            self.assertTrue(opened["open"], opened)
            self.assertFalse(opened["simulation_running"], opened)
            self.assertEqual(opened["phase"], "paused")
            self.assertEqual(opened["scene_label"], "Empty stage")
            self.assertTrue(opened["viewer_url"].startswith("http://127.0.0.1:"))
            workspace = runtime.get_session(runtime.WORKSPACE_RUN_ID)
            self.assertIsNotNone(workspace)
            plane_handles = _wait_for(
                lambda: dict(workspace.viewer._viewer._plane_handles),
                5.0,
                "Empty Newton workspace did not publish its ground grid",
            )
            handle_ids = {
                name: tuple(id(handle) for handle in handles)
                for name, handles in plane_handles.items()
            }
            time.sleep(0.2)
            stable_handle_ids = {
                name: tuple(id(handle) for handle in handles)
                for name, handles in workspace.viewer._viewer._plane_handles.items()
            }
            self.assertEqual(stable_handle_ids, handle_ids)
            self.assertTrue(workspace.viewer._plane_rebuild_counts)
            self.assertTrue(all(count == 1 for count in workspace.viewer._plane_rebuild_counts.values()))

            playing = runtime.control_workspace("play")
            self.assertTrue(playing["simulation_running"], playing)
            advanced = _wait_for(
                lambda: runtime.control_workspace("status")
                if runtime.control_workspace("status")["frame_count"] >= 2
                else None,
                5.0,
                "Empty Newton workspace did not advance physics",
            )
            self.assertFalse(advanced["last_error"], advanced)

            stopped = runtime.control_workspace("stop")
            self.assertTrue(stopped["open"], stopped)
            self.assertFalse(stopped["simulation_running"], stopped)
            self.assertEqual(stopped["phase"], "paused")
        finally:
            closed = runtime.control_workspace("close")
            self.assertFalse(closed["open"], closed)


if __name__ == "__main__":
    unittest.main()
