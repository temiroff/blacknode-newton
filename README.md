# blacknode-newton

`blacknode-newton` provides a Newton simulation workspace inside the Blacknode editor. **Open Newton** starts a real empty stage with a ground grid, an attached or floating viewport, File and Simulation menus, and stopped physics. USD scenes can be opened and operated directly in that workspace. Workflow nodes remain available for automation, ROS connectivity, recording, and policy execution.

## What is included

- Newton 1.4 XPBD simulation on CUDA or CPU.
- The supplied `so101_robot.usd` as a self-contained example asset with attribution; any local or package USD can be selected.
- A browser viewer and teleoperation panel powered by Viser, embedded above the Blacknode canvas. Its scene/control traffic uses the viewer's WebSocket server.
- An optional experimental NVIDIA OVRT 0.4 viewer that renders the same USD/Newton session with RTX in an isolated process and streams JPEG frames into the same dockable Blacknode surface.
- Articulation controls discovered from USD revolute and prismatic joints with finite limits.
- Explicit arm/disarm, pause, reset, stop, joint-limit clamping, and velocity/step limiting.
- Authored USD collision meshes, optional ground contact, and optional box rigid bodies. Authored fixed and free rigid-body joints are preserved, collision schemas are normalized in memory, and the source file is left unchanged.
- An optional `NewtonROSBridge` component using `sensor_msgs/msg/JointState` through `rosbridge_websocket`.

## Open a USD scene

1. In the Blacknode editor, open **Packages**, select `blacknode-newton`, and press **Install prerequisites**. The default `runtime` and `viewer-viser` components must be enabled.
2. Press **Open Newton**. The workspace opens above the node canvas with an empty stage and stopped physics. Drag the divider to resize it, or use the View menu to float and attach it inside the Blacknode window.
3. In the Newton app bar choose **File → Open USD…**. Blacknode's in-editor file browser accepts `.usd`, `.usda`, and `.usdc`; opening a file replaces the workspace stage and leaves simulation stopped at the initial state.
4. Use **Simulation → Start**, **Stop**, and **Reset**, or the matching transport buttons. Stop pauses physics while keeping the stage and viewer open. **File → New stage** returns to the empty stage and **Close Newton** shuts down the workspace service.
5. Check **Arm joint commands** in the Viser panel before moving articulation sliders. Uncheck it before changing tasks.

### Try the OVRT renderer

The `viewer-ovrtx` component is separate and disabled by default because its pinned OVRT/OVStage release train is a large optional RTX dependency. In **Packages**, enable `viewer-ovrtx` for `blacknode-newton` and install that component's prerequisites. The Newton View menu then exposes **Open with OVRT (RTX)** and **Use OVRT (RTX) renderer**. Switching renderer restarts the current generic scene; it does not create a renderer-specific graph or scene format.

OVRT runs in a child process while Blacknode hosts the responsive embedded page. Newton body poses are written into OVStage at monotonically increasing ordinals, OVRT reads `LdrColor`, and the page receives a loopback HTTP MJPEG stream. The composed render stage preserves the selected USD's authored units and Y-up or Z-up axis, and its initial camera frames authored render bounds. An empty stage opens on a visible metre-scale grid with colored principal axes, making renderer readiness clear before an asset is selected. Left drag orbits, right or middle drag pans, the wheel zooms, and double-click restores the initial camera; native browser image dragging is disabled and pointer motion is coalesced once per display frame. The process starts only after the browser proxy is available, so the viewer can show native renderer initialization and shader compilation progress. A newly encountered RTX material/shader set can make the first usable frame slow; initialization publishes the first valid `LdrColor` frame immediately.

The current OVRT surface is an experimental renderer qualification. Simulation start/stop/reset remains in the Newton app bar, and articulation commands continue to enter the managed Newton session through workflows or the ROS bridge. Viser remains the default when its in-viewer joint sliders and HDRI controls are needed. OVRT requires a supported NVIDIA RTX GPU and driver; its component reports unavailable independently when the optional native packages are absent.

The workspace is a managed Newton service and does not create or modify graph nodes. `NewtonUSDScene`, `NewtonViewerConfig`, `NewtonSimulation`, and `NewtonJointCommand` remain the typed workflow interface when a graph needs to create a separately configured scene, control lifecycle, or automate joint targets. The editor workspace and workflow sessions use distinct run identifiers.

Visual-only USD files are accepted and displayed. Their status reports zero authored dynamic bodies and a warning that collision geometry is absent. Physics motion requires authored rigid bodies/colliders or explicit bodies supplied by a workflow scene.

Open the viewer's **View** folder to change the solid background, select a Viser HDRI preset and its intensity, show or hide the HDRI background, truly suppress Newton's physics grid, set camera position and orbit target, select the world up axis, and change keyboard translation speed. The Viser provider keeps unchanged plane handles alive across frames, so a stationary physics grid is not removed and recreated by the renderer. **Apply camera** uses the entered values; **Frame scene** restores an automatic view. Left-drag orbits around the displayed target, right-drag pans, and scrolling zooms.

Viser 1.x exposes its bundled HDRI presets (`apartment`, `city`, `dawn`, `forest`, `lobby`, `night`, `park`, `studio`, `sunset`, and `warehouse`) through the viewer API. `hdri=none` uses the selected solid background. A future viewer provider can implement the same environment contract with its own local-file environment-map loader.

The scene status reports `authored_dynamic_body_names`, `authored_dynamic_body_count`, mesh/normal counts, and collision-mesh counts. A USD that contains only visual meshes has no physical bodies to move; author `RigidBodyAPI`, collision geometry, mass, and a free joint in the asset, or add an explicit rigid body through `NewtonUSDScene.rigid_bodies`.

`NewtonSimulation` exposes Newton's position-drive gains as `joint_stiffness` and `joint_damping`. The defaults are `1e8` and `1e5`, qualified for this XPBD setup at 60 Hz with four substeps. `joint_drive_overrides` can replace either value for named joints, for example:

```json
{
  "gripper": {"stiffness": 50000000.0, "damping": 50000.0}
}
```

Lower stiffness gives more compliance; damping resists oscillation. Stop and restart the Newton session after changing these build-time values. The applied per-joint values are reported in `NewtonSimulation.session.joint_drive_gains`.

The configured viewer port is a preference. If another Windows application already owns that loopback port, the provider selects a free port and reports the actual URL through `NewtonSimulation.viewer_url`.

The first CUDA run compiles and caches Warp contact kernels. Convex decomposition runs only for path patterns explicitly listed in the scene node.

## Windows and ROS 2

The viewer works directly on Windows: Viser serves the page and streams the scene over WebSocket from the Blacknode process. ROS 2 is a separate transport boundary.

To connect a ROS graph from Windows:

1. Run `rosbridge_websocket` on a machine or container with ROS 2, reachable on TCP port 9090.
2. Enable `blacknode-ros2/rosbridge`, then enable the optional `blacknode-newton/rosbridge` component in **Packages**.
3. Start the Newton simulation, then start `NewtonROSBridge` with the rosbridge host.
4. Read `/blacknode/newton/joint_states` and publish named SI-unit targets to `/blacknode/newton/joint_commands` as `sensor_msgs/msg/JointState` (radians for revolute joints and metres for prismatic joints).

To mirror a real robot into Newton, set `command_topic` to the real robot's `JointState` topic (commonly `/joint_states`) and keep `state_topic` on a separate Newton-only topic. Incoming joint names must match the USD articulation names exactly. This path is real robot state → Newton simulation only; it does not command the physical robot.

The simulator still starts disarmed. ROS commands are rejected until the corresponding Newton session is explicitly armed. After a ROS command is accepted, the bridge disarms the session if that command stream is stale for more than `command_stale_seconds` (default `0.5 s`).

The Viser server listens on all interfaces so another machine can use the host's LAN address instead of `127.0.0.1`. Treat the port as an operator control endpoint: expose it only on a trusted network or behind an authenticated reverse proxy.

## Component boundary

The Newton session owns physics, state, limits, and the command safety gate. Viewer providers implement the small contract in `nodes/viewer_contract.py`; `viewer-viser` and the optional `viewer-ovrtx` register independently, while transports call the managed session API. This keeps rendering and ROS connectivity replaceable while the workflow's scene and command contracts remain stable.

See [docs/FUTURE_DEVELOPMENT.md](docs/FUTURE_DEVELOPMENT.md) for the separately scoped OVUI, recording, policy replay, and reinforcement-learning plan.

## Development

From this package worktree:

```powershell
$env:PYTHONPATH = "..\..\python"
python -m unittest discover -s tests -v
```

The package is licensed under Apache-2.0. Model attribution is in `NOTICE`.
