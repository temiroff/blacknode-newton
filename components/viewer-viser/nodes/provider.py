"""Viser browser/WebSocket implementation of the Newton viewer contract."""
from __future__ import annotations

import math
import socket
import types
from collections import deque
from typing import Any

from blacknode.pkg.blacknode_newton.viewer_contract import register_viewer


def _viser_hdri(value: Any) -> str:
    hdri = str(value or "none").lower()
    return hdri if hdri in {
        "none", "apartment", "city", "dawn", "forest", "lobby", "night",
        "park", "studio", "sunset", "warehouse",
    } else "none"


def _loopback_port_available(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _select_viewer_port(requested: int) -> int:
    requested = max(1024, min(65535, int(requested)))
    if _loopback_port_available(requested):
        return requested
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


class ViserTeleoperationViewer:
    def __init__(self, session: Any, model: Any, config: dict[str, Any]) -> None:
        try:
            from newton.viewer import ViewerViser
        except Exception as exc:  # pragma: no cover - package health catches this
            raise RuntimeError("Newton's Viser viewer and viser>=1 are required") from exc
        self.session = session
        self.requested_port = int(config.get("port") or 8080)
        self.port = _select_viewer_port(self.requested_port)
        self._viewer = ViewerViser(
            port=self.port,
            label=str(config.get("label") or "Blacknode Newton"),
            verbose=False,
            share=False,
        )
        self._simulation_layer = "simulation"
        self._reference_layer = "real_reference"
        self._reference_options: dict[str, Any] = {}
        self._ghost_batch_masks: dict[str, Any] = {}
        self._viewer.activate(self._simulation_layer)
        self._viewer.show_visual = bool(config.get("show_visuals", True))
        self._viewer.show_collision = bool(config.get("show_colliders", False))
        self._viewer.show_particles = bool(getattr(model, "particle_count", 0))
        self._viewer.set_model(model)
        self._viewer.activate(self._reference_layer)
        self._viewer.show_visual = True
        self._viewer.show_collision = False
        self._viewer.show_particles = False
        self._viewer.show_static = False
        self._viewer.set_model(model)
        self._viewer.set_layer_visible(self._reference_layer, False)
        self._prepare_reference_layer(model)
        self._viewer.activate(self._simulation_layer)
        self._background_rgb = self._parse_color(str(config.get("background_color") or "#6383c5"))
        self._grid_visible = bool(config.get("show_grid", True))
        environment = dict(config.get("environment") or {})
        self._hdri = _viser_hdri(environment.get("hdri"))
        self._hdri_enabled = bool(environment.get("hdri_enabled", True))
        self._show_hdri_background = bool(environment.get("show_background", True))
        self._hdri_intensity = float(environment.get("intensity", 1.0))
        self._plane_signatures: dict[str, tuple[Any, ...]] = {}
        self._plane_rebuild_counts: dict[str, int] = {}
        self._install_grid_filter()
        self._install_reference_style()
        self._install_particle_shader()
        camera = dict(config.get("camera") or {})
        self._camera_up_axis = str(camera.get("up_axis") or "auto").lower()
        self._viewer.camera_speed = float(camera.get("speed_m_s") or 1.0)
        self._apply_environment()
        position = list(camera.get("position_m") or [])
        target = list(camera.get("target_m") or [])
        if len(position) == 3 and len(target) == 3:
            self._set_camera_view(position, target, self._camera_up_axis)
        else:
            self._frame_scene(self._camera_up_axis)
        gui = self._viewer._server.gui  # ViewerViser intentionally exposes no GUI facade.
        self._status = gui.add_markdown("**Simulation motion disarmed** · starting")
        with gui.add_folder("View"):
            background = gui.add_rgb("Background", initial_value=self._background_rgb)
            show_grid = gui.add_checkbox("Show physics grid", initial_value=self._grid_visible)
            hdri = gui.add_dropdown(
                "HDRI preset",
                options=(
                    "none", "apartment", "city", "dawn", "forest", "lobby", "night",
                    "park", "studio", "sunset", "warehouse",
                ),
                initial_value=self._hdri,
            )
            show_hdri_background = gui.add_checkbox(
                "Show HDRI background", initial_value=self._show_hdri_background,
            )
            hdri_intensity = gui.add_number(
                "HDRI intensity", initial_value=self._hdri_intensity,
                min=0.0, max=10.0, step=0.1,
            )
            camera_position = gui.add_vector3(
                "Camera position (m)", initial_value=tuple(self._camera_view[0]), step=0.01,
            )
            camera_target = gui.add_vector3(
                "Orbit target (m)", initial_value=tuple(self._camera_view[1]), step=0.01,
            )
            camera_up_axis = gui.add_dropdown(
                "Up axis", options=("auto", "X", "Y", "Z"),
                initial_value=self._camera_up_axis if self._camera_up_axis == "auto" else self._camera_up_axis.upper(),
            )
            camera_speed = gui.add_number(
                "Keyboard speed (m/s)", initial_value=self._viewer.camera_speed,
                min=0.0, max=100.0, step=0.1,
            )
            apply_camera = gui.add_button("Apply camera")
            frame_scene = gui.add_button("Frame scene")
        with gui.add_folder("Safety and simulation"):
            arm = gui.add_checkbox("Arm joint commands", initial_value=False)
            pause = gui.add_checkbox("Pause physics", initial_value=False)
            reset = gui.add_button("Reset scene")
        with gui.add_folder("Articulation joint targets"):
            sliders = {}
            for name in session.joint_indices:
                lower, upper = session.joint_limits[name]
                angular = session.joint_units.get(name) == "radians"
                sliders[name] = gui.add_slider(
                    f"{name.replace('_', ' ').title()} ({'deg' if angular else 'm'})",
                    min=math.degrees(lower) if angular else lower,
                    max=math.degrees(upper) if angular else upper,
                    step=0.1 if angular else max(0.0001, (upper - lower) / 1000.0),
                    initial_value=math.degrees(session.desired[name]) if angular else session.desired[name],
                )

        @arm.on_update
        def _arm_changed(event: Any) -> None:
            try:
                session.set_armed(bool(event.target.value))
            except Exception:
                event.target.value = False

        @pause.on_update
        def _pause_changed(event: Any) -> None:
            session.set_paused(bool(event.target.value))

        @reset.on_click
        def _reset_clicked(_event: Any) -> None:
            arm.value = False
            session.request_reset()
            for name, slider in sliders.items():
                value = session.home[name]
                slider.value = math.degrees(value) if session.joint_units.get(name) == "radians" else value

        @background.on_update
        def _background_changed(event: Any) -> None:
            self._background_rgb = tuple(int(value) for value in event.target.value)
            self._apply_environment()

        @show_grid.on_update
        def _grid_changed(event: Any) -> None:
            self._grid_visible = bool(event.target.value)
            self._set_plane_visibility()

        def _environment_changed(_event: Any) -> None:
            self._hdri = str(hdri.value)
            self._show_hdri_background = bool(show_hdri_background.value)
            self._hdri_intensity = float(hdri_intensity.value)
            self._apply_environment()

        hdri.on_update(_environment_changed)
        show_hdri_background.on_update(_environment_changed)
        hdri_intensity.on_update(_environment_changed)

        @camera_speed.on_update
        def _camera_speed_changed(event: Any) -> None:
            self._viewer.camera_speed = float(event.target.value)

        @apply_camera.on_click
        def _apply_camera_clicked(_event: Any) -> None:
            self._camera_up_axis = str(camera_up_axis.value).lower()
            self._set_camera_view(camera_position.value, camera_target.value, self._camera_up_axis)

        @frame_scene.on_click
        def _frame_scene_clicked(_event: Any) -> None:
            self._camera_up_axis = str(camera_up_axis.value).lower()
            self._frame_scene(self._camera_up_axis)
            camera_position.value = tuple(self._camera_view[0])
            camera_target.value = tuple(self._camera_view[1])

        for joint_name, slider in sliders.items():
            def _target_changed(event: Any, name: str = joint_name) -> None:
                if not session.armed:
                    return
                try:
                    value = float(event.target.value)
                    if session.joint_units.get(name) == "radians":
                        value = math.radians(value)
                    session.command({name: value}, source="viser")
                except Exception:
                    pass

            slider.on_update(_target_changed)

    def _scene_bounds(self) -> tuple[list[float], float]:
        positions: list[list[float]] = []
        try:
            for transform in self.session.state_0.body_q.numpy():
                point = [float(value) for value in transform[:3]]
                if all(math.isfinite(value) for value in point):
                    positions.append(point)
        except Exception:
            positions = []
        for point in dict(self.session.status().get("rigid_body_positions_m") or {}).values():
            values = [float(value) for value in list(point)[:3]]
            if len(values) == 3 and all(math.isfinite(value) for value in values):
                positions.append(values)
        if not positions:
            return [0.0, 0.0, 0.15], 0.5
        lower = [min(point[axis] for point in positions) for axis in range(3)]
        upper = [max(point[axis] for point in positions) for axis in range(3)]
        center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
        body_extent = max(upper[axis] - lower[axis] for axis in range(3))
        shape_scale = float(getattr(self._viewer, "scene_scale", 0.0) or 0.0)
        return center, max(0.35, body_extent, shape_scale * 6.0)

    @staticmethod
    def _parse_color(value: str) -> tuple[int, int, int]:
        value = value.strip().lstrip("#")
        if len(value) != 6 or any(character not in "0123456789abcdefABCDEF" for character in value):
            return (17, 24, 39)
        return tuple(int(value[offset:offset + 2], 16) for offset in (0, 2, 4))

    def _set_background(self, color: tuple[int, int, int]) -> None:
        import numpy as np

        image = np.empty((2, 2, 3), dtype=np.uint8)
        image[:, :] = color
        self._viewer._server.scene.set_background_image(image, format="png")

    def _apply_environment(self) -> None:
        hdri = None if self._hdri == "none" or not self._hdri_enabled else self._hdri
        show_background = hdri is not None and self._show_hdri_background
        self._viewer._server.scene.configure_environment_map(
            hdri,
            background=show_background,
            background_intensity=self._hdri_intensity,
            environment_intensity=self._hdri_intensity,
        )
        if show_background:
            self._viewer._server.scene.set_background_image(None)
        else:
            self._set_background(self._background_rgb)

    def _install_grid_filter(self) -> None:
        original = self._viewer._log_plane_instances

        def _signature(plane_info: dict[str, Any], xforms: Any, scales: Any) -> tuple[Any, ...]:
            import numpy as np

            xforms_array = self._viewer._to_numpy(xforms)
            scales_array = self._viewer._to_numpy(scales) if scales is not None else None
            xforms_np = np.asarray(xforms_array) if xforms_array is not None else np.empty((0,))
            scales_np = np.asarray(scales_array) if scales_array is not None else np.empty((0,))
            return (
                tuple(sorted((str(key), float(value)) for key, value in plane_info.items())),
                xforms_np.shape,
                xforms_np.tobytes(),
                scales_np.shape,
                scales_np.tobytes(),
            )

        def _log_plane_instances(
            _viewer: Any,
            name: str,
            plane_info: dict[str, Any],
            xforms: Any,
            scales: Any,
            hidden: bool = False,
        ) -> None:
            should_hide = bool(hidden) or not self._grid_visible
            if should_hide:
                self._plane_signatures.pop(name, None)
                original(name, plane_info, xforms, scales, hidden=True)
                return
            signature = _signature(plane_info, xforms, scales)
            handles = self._viewer._plane_handles.get(name)
            if handles and self._plane_signatures.get(name) == signature:
                return
            original(name, plane_info, xforms, scales, hidden=False)
            self._plane_signatures[name] = signature
            self._plane_rebuild_counts[name] = self._plane_rebuild_counts.get(name, 0) + 1

        self._viewer._log_plane_instances = types.MethodType(
            _log_plane_instances, self._viewer
        )

    def _prepare_reference_layer(self, model: Any) -> None:
        import numpy as np

        robot_bodies = set(getattr(self.session, "articulation_body_indices", set()) or set())
        shape_bodies = model.shape_body.numpy().tolist()
        for batch in self._viewer._shape_instances.values():
            qualified = str(batch.name)
            self._ghost_batch_masks[qualified] = np.asarray(
                [
                    1.0 if int(shape_bodies[int(shape_index)]) in robot_bodies else 0.0
                    for shape_index in batch.model_shapes
                ],
                dtype=np.float32,
            )
        # Ghosts use a uniform diagnostic material rather than duplicating
        # source textures, which keeps opacity available for every mesh batch.
        for mesh in self._viewer._meshes.values():
            mesh["texture"] = None
            mesh["trimesh"] = None

    def _install_reference_style(self) -> None:
        original = self._viewer.log_instances

        def _log_instances(
            viewer: Any,
            name: str,
            mesh: str,
            xforms: Any,
            scales: Any,
            colors: Any,
            materials: Any,
            hidden: bool = False,
        ) -> None:
            original(name, mesh, xforms, scales, colors, materials, hidden=hidden)
            if viewer._active_layer_id != self._reference_layer or hidden:
                return
            import numpy as np

            qualified = viewer._qualify(name)
            handle = viewer._scene_handles.get(qualified)
            mask = self._ghost_batch_masks.get(qualified)
            if handle is None or mask is None:
                return
            color = list(self._reference_options.get("color_rgb") or [0.18, 0.86, 1.0])
            opacity = float(self._reference_options.get("opacity") or 0.28)
            count = len(mask)
            if hasattr(handle, "batched_colors"):
                handle.batched_colors = np.tile(
                    np.asarray(color, dtype=np.float32).clip(0.0, 1.0) * 255.0,
                    (count, 1),
                ).astype(np.uint8)
            if hasattr(handle, "batched_opacities"):
                handle.batched_opacities = mask * max(0.05, min(0.95, opacity))
            if hasattr(handle, "cast_shadow"):
                handle.cast_shadow = False

        self._viewer.log_instances = types.MethodType(_log_instances, self._viewer)

    def _install_particle_shader(self) -> None:
        """Render grains through one stable, shaded float32 point buffer."""
        original = self._viewer.log_points
        particle_path = f"/layers/{self._simulation_layer}/model/particles"
        fill = dict(self.session.scene.get("particle_fill") or {})
        base_color = list(fill.get("color_rgb") or [0.82, 0.58, 0.18])

        def _log_points(
            viewer: Any,
            name: str,
            points: Any,
            radii: Any = None,
            colors: Any = None,
            hidden: bool = False,
        ) -> None:
            qualified = viewer._qualify(name)
            if qualified != particle_path:
                original(name, points, radii, colors, hidden=hidden)
                return

            import numpy as np

            handle = viewer._scene_handles.get(qualified)
            if hidden or points is None:
                if handle is not None:
                    handle.visible = False
                return
            positions = viewer._to_numpy(points).astype(np.float32, copy=False)
            if not len(positions):
                if handle is not None:
                    handle.visible = False
                return
            point_size = float(np.mean(viewer._to_numpy(radii))) if radii is not None else 0.001
            rgb = np.clip(np.asarray(base_color, dtype=np.float32), 0.0, 1.0)
            grain_colors = np.tile((rgb * 255.0).astype(np.uint8), (len(positions), 1))
            if handle is None:
                handle = viewer._server.scene.add_point_cloud(
                    name=qualified,
                    points=positions,
                    colors=grain_colors,
                    point_size=point_size,
                    point_shape="circle",
                    point_shading="gradient",
                    precision="float32",
                )
                viewer._scene_handles[qualified] = handle
                return
            handle.points = positions
            if not handle.visible:
                handle.visible = True

        self._viewer.log_points = types.MethodType(_log_points, self._viewer)

    def _up_axis_index(self, value: str) -> int:
        return {"x": 0, "y": 1, "z": 2}.get(str(value).lower(), int(self._viewer._get_camera_up_axis()))

    def _set_camera_view(self, position: Any, target: Any, up_axis: str) -> None:
        import numpy as np

        position_array = np.asarray(position, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if position_array.shape != (3,) or target_array.shape != (3,):
            raise ValueError("camera position and target must contain exactly three values")
        direction = target_array - position_array
        if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(target_array)):
            raise ValueError("camera position and target must be finite")
        if float(np.linalg.norm(direction)) < 1.0e-6:
            raise ValueError("camera position and target must be different")
        up_direction = np.zeros(3, dtype=np.float64)
        up_direction[self._up_axis_index(up_axis)] = 1.0
        if float(np.linalg.norm(np.cross(direction, up_direction))) < 1.0e-6:
            up_direction = np.array((0.0, 1.0, 0.0) if up_direction[1] == 0.0 else (0.0, 0.0, 1.0))
        self._camera_view = (position_array.tolist(), target_array.tolist())
        self._viewer._camera_request = (position_array, target_array, up_direction)
        initial_camera = self._viewer._server.initial_camera
        initial_camera.position = tuple(position_array.tolist())
        initial_camera.look_at = tuple(target_array.tolist())
        initial_camera.up = tuple(up_direction.tolist())
        for client in self._viewer._server.get_clients().values():
            self._viewer._apply_camera_to_client(client)

    def _frame_scene(self, up_axis_name: str = "auto") -> None:
        center, extent = self._scene_bounds()
        distance = max(0.7, extent * 1.8)
        up_axis = self._up_axis_index(up_axis_name)
        if up_axis == 0:
            position = [center[0] + distance * 0.7, center[1] - distance, center[2] + distance]
        elif up_axis == 1:
            position = [center[0] + distance, center[1] + distance * 0.7, center[2] + distance]
        else:
            position = [center[0] + distance, center[1] - distance, center[2] + distance * 0.7]
        self._set_camera_view(position, center, up_axis_name)

    def _set_plane_visibility(self) -> None:
        for value in self._viewer._plane_handles.values():
            handles = value if isinstance(value, (list, tuple)) else (value,)
            for handle in handles:
                handle.visible = self._grid_visible

    @property
    def url(self) -> str:
        return self._viewer.url.replace("localhost", "127.0.0.1")

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def begin_frame(self, time_seconds: float) -> None:
        self._viewer.begin_frame(time_seconds)

    def log_state(self, state: Any) -> None:
        self._viewer.activate(self._simulation_layer)
        self._viewer.log_state(state)

    def log_reference_state(self, state: Any | None, options: dict[str, Any]) -> None:
        self._reference_options = dict(options)
        visible = bool(state is not None and options.get("visible", True))
        self._viewer.set_layer_visible(self._reference_layer, visible)
        if not visible:
            self._viewer.activate(self._simulation_layer)
            return
        self._viewer.set_layer_transform(
            self._reference_layer, list(options.get("offset_m") or [0.0, 0.0, 0.0])
        )
        self._viewer.activate(self._reference_layer)
        self._viewer.log_state(state)
        self._viewer.activate(self._simulation_layer)

    def end_frame(self) -> None:
        self._viewer.end_frame()
        self._set_plane_visibility()
        status = self.session.status()
        mode = "ARMED" if status["armed"] else "Simulation motion disarmed"
        bodies = dict(status.get("rigid_body_positions_m") or {})
        body_text = " · ".join(
            f"{name}: {', '.join(f'{value:.3f}' for value in position)}"
            for name, position in list(bodies.items())[:3]
        ) or "USD scene"
        self._status.content = (
            f"**{mode}** · {status['phase']} · frame {status['frame_count']}  \n"
            f"Rigid bodies (m): `{body_text}`"
        )

    def set_visibility(self, path: str, visible: bool) -> bool:
        # The empty workspace exposes its generated ground as one outliner
        # object. Viser can update that plane collection in place; imported USD
        # prims are flattened into Newton shape batches and have no stable prim
        # path mapping in this provider.
        if str(path) != "/Blacknode/Ground":
            return False
        self._viewer.activate(self._simulation_layer)
        self._grid_visible = bool(visible)
        self._set_plane_visibility()
        return True

    def set_grid(self, visible: bool) -> bool:
        self._viewer.activate(self._simulation_layer)
        self._grid_visible = bool(visible)
        self._set_plane_visibility()
        return True

    def set_render_options(self, show_visuals: bool, show_colliders: bool) -> bool:
        self._viewer.activate(self._simulation_layer)
        self._viewer.show_visual = bool(show_visuals)
        self._viewer.show_collision = bool(show_colliders)
        return True

    def set_transform(self, path: str, transform: dict[str, Any]) -> bool:
        del path, transform
        return False

    def set_material(self, path: str, material_path: str, material: dict[str, Any]) -> bool:
        del path, material_path, material
        return False

    def set_environment(self, environment: dict[str, Any]) -> bool:
        self._background_rgb = self._parse_color(
            str(environment.get("background_color") or "#6383c5")
        )
        self._hdri = _viser_hdri(environment.get("hdri"))
        self._hdri_enabled = bool(environment.get("hdri_enabled", True))
        self._show_hdri_background = bool(environment.get("show_background", True))
        self._hdri_intensity = float(environment.get("intensity", 1.0))
        self._apply_environment()
        return True

    def close(self) -> None:
        self._viewer.close()


class ViserTrainingViewer:
    """Read-only view of one sampled articulation from a batched RL job."""

    def __init__(self, environment: Any, model: Any, config: dict[str, Any]) -> None:
        try:
            import numpy as np
            from newton.viewer import ViewerViser
        except Exception as exc:  # pragma: no cover - package health catches this
            raise RuntimeError("Newton's Viser viewer and viser>=1 are required") from exc
        self._np = np
        self.environment = environment
        self.requested_port = int(config.get("port") or 8091)
        self.port = _select_viewer_port(self.requested_port)
        self._viewer = ViewerViser(
            port=self.port,
            label=str(config.get("label") or "SO-ARM101 PPO Training"),
            verbose=False,
            share=False,
        )
        self._viewer.show_visual = bool(config.get("show_visuals", True))
        self._viewer.show_collision = False
        self._viewer.show_particles = False
        self._viewer.set_model(model)
        server = self._viewer._server
        background = ViserTeleoperationViewer._parse_color(
            str(config.get("background_color") or "#111827")
        )
        image = np.empty((2, 2, 3), dtype=np.uint8)
        image[:, :] = background
        server.scene.set_background_image(image, format="png")
        self._grid = server.scene.add_grid(
            "/blacknode_training/grid", width=1.4, height=1.4, plane="xy",
            cell_size=0.05, section_size=0.25, cell_color=(80, 92, 115),
            section_color=(129, 140, 166), plane_opacity=0.04,
        )
        self._target = server.scene.add_icosphere(
            "/blacknode_training/target", radius=0.025, color=(34, 197, 94),
            subdivisions=2, material="toon3", cast_shadow=False,
        )
        self._trail = server.scene.add_line_segments(
            "/blacknode_training/end_effector_trail",
            points=np.empty((0, 2, 3), dtype=np.float32),
            colors=(251, 146, 60), line_width=3.0,
        )
        self._trail_points: deque[tuple[float, float, float]] = deque(maxlen=240)
        self._last_target: tuple[float, float, float] | None = None
        self._show_trail = True
        self._show_target = True
        self._set_camera()
        gui = server.gui
        self._status = gui.add_markdown(
            "**PPO starting** · sampled simulation only · physical motion disarmed"
        )
        with gui.add_folder("Training preview"):
            environment_index = gui.add_slider(
                "Environment index", min=0,
                max=max(0, int(environment.environment_count) - 1), step=1,
                initial_value=int(environment.preview_environment_index),
            )
            show_target = gui.add_checkbox("Show target", initial_value=True)
            show_trail = gui.add_checkbox("Show end-effector trail", initial_value=True)
            reset_camera = gui.add_button("Frame SO-ARM101")

        @environment_index.on_update
        def _environment_changed(event: Any) -> None:
            environment.set_preview_environment(int(event.target.value))
            self._trail_points.clear()
            self._last_target = None

        @show_target.on_update
        def _target_changed(event: Any) -> None:
            self._show_target = bool(event.target.value)
            self._target.visible = self._show_target

        @show_trail.on_update
        def _trail_changed(event: Any) -> None:
            self._show_trail = bool(event.target.value)
            self._trail.visible = self._show_trail and len(self._trail_points) > 1

        @reset_camera.on_click
        def _reset_camera(_event: Any) -> None:
            self._set_camera()

    def _set_camera(self) -> None:
        np = self._np
        position = np.asarray((0.72, -0.72, 0.56), dtype=np.float64)
        target = np.asarray((0.0, 0.0, 0.22), dtype=np.float64)
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        self._viewer._camera_request = (position, target, up)
        camera = self._viewer._server.initial_camera
        camera.position = tuple(position.tolist())
        camera.look_at = tuple(target.tolist())
        if hasattr(camera, "up"):
            camera.up = tuple(up.tolist())
        else:
            camera.up_direction = tuple(up.tolist())
        for client in self._viewer._server.get_clients().values():
            self._viewer._apply_camera_to_client(client)

    @property
    def url(self) -> str:
        return self._viewer.url.replace("localhost", "127.0.0.1")

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def begin_frame(self, time_seconds: float) -> None:
        self._viewer.begin_frame(time_seconds)

    def log_state(self, state: Any) -> None:
        self._viewer.log_state(state)

    def log_training_state(self, state: Any, metadata: dict[str, Any]) -> None:
        np = self._np
        self._viewer.log_state(state)
        target = tuple(float(value) for value in list(metadata.get("target_m") or [])[:3])
        end_effector = tuple(
            float(value) for value in list(metadata.get("end_effector_m") or [])[:3]
        )
        if len(target) == 3:
            self._target.position = target
            self._target.visible = self._show_target
            if self._last_target is None or max(
                abs(left - right) for left, right in zip(target, self._last_target)
            ) > 1e-5:
                self._trail_points.clear()
                self._last_target = target
        if len(end_effector) == 3:
            if not self._trail_points or max(
                abs(left - right) for left, right in zip(end_effector, self._trail_points[-1])
            ) > 1e-5:
                self._trail_points.append(end_effector)
        if len(self._trail_points) > 1:
            points = np.asarray(self._trail_points, dtype=np.float32)
            self._trail.points = np.stack((points[:-1], points[1:]), axis=1)
            self._trail.visible = self._show_trail
        else:
            self._trail.visible = False
        success = " · **SUCCESS**" if bool(metadata.get("success")) else ""
        self._status.content = (
            f"**SO-ARM101 PPO** · update {int(metadata.get('update') or 0)}/"
            f"{int(metadata.get('updates') or 0)} · environment "
            f"{int(metadata.get('environment_index') or 0)}  \n"
            f"distance `{float(metadata.get('distance_m') or 0.0):.4f} m` · "
            f"reward `{float(metadata.get('reward') or 0.0):.3f}` · "
            f"step {int(metadata.get('episode_step') or 0)}{success}  \n"
            "Sampled Newton/Warp simulation · **physical motion disarmed**"
        )

    def log_reference_state(self, state: Any | None, options: dict[str, Any]) -> None:
        del state, options

    def end_frame(self) -> None:
        self._viewer.end_frame()

    def set_visibility(self, path: str, visible: bool) -> bool:
        del path, visible
        return False

    def set_grid(self, visible: bool) -> bool:
        self._grid.visible = bool(visible)
        return True

    def set_render_options(self, show_visuals: bool, show_colliders: bool) -> bool:
        self._viewer.show_visual = bool(show_visuals)
        self._viewer.show_collision = bool(show_colliders)
        return True

    def set_transform(self, path: str, transform: dict[str, Any]) -> bool:
        del path, transform
        return False

    def set_material(self, path: str, material_path: str, material: dict[str, Any]) -> bool:
        del path, material_path, material
        return False

    def set_environment(self, environment: dict[str, Any]) -> bool:
        del environment
        return False

    def close(self) -> None:
        self._viewer.close()


def _factory(session: Any, model: Any, config: dict[str, Any]) -> Any:
    if str(config.get("mode") or "").lower() == "training-preview":
        return ViserTrainingViewer(session, model, config)
    return ViserTeleoperationViewer(session, model, config)


register_viewer("viser", _factory)
