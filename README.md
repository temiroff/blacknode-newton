# blacknode-newton

`blacknode-newton` provides a Newton simulation workspace inside the Blacknode editor. **Open Newton** loads the bundled SO-101 tabletop scene with stopped physics, a hidden reference grid, and an attached or floating viewport. USD scenes can be opened and operated directly in that workspace. Workflow nodes remain available for automation, ROS connectivity, recording, and policy execution.

## What is included

- Newton 1.4 XPBD simulation on CUDA or CPU.
- The supplied `so101_robot.usd` robot and `scenes/so101_tabletop.usd` pick scene as self-contained example assets with attribution; local USD, URDF, Xacro, and MuJoCo MJCF robot descriptions can be selected.
- A browser viewer and teleoperation panel powered by Viser, embedded above the Blacknode canvas. Its scene/control traffic uses the viewer's WebSocket server.
- An optional experimental NVIDIA OVRT 0.4 viewer that renders the same USD/Newton session with RTX in an isolated process and streams JPEG frames into the same dockable Blacknode surface.
- Articulation controls discovered from USD revolute and prismatic joints with finite limits.
- Explicit arm/disarm, pause, reset, stop, joint-limit clamping, and velocity/step limiting.
- Authored USD collision meshes, optional ground contact, and optional box rigid bodies. Authored fixed and free rigid-body joints are preserved, collision schemas are normalized in memory, and the source file is left unchanged.
- An optional `NewtonROSBridge` component using `sensor_msgs/msg/JointState` through `rosbridge_websocket`.
- An optional `NewtonReplayBridge` component that maps recorded Dataset Browser episodes onto the robot already loaded in the Newton workspace.

## Open a scene

1. In the Blacknode editor, open **Packages**, select `blacknode-newton`, and press **Install prerequisites**. The default `runtime` and `viewer-viser` components must be enabled.
2. Press **Open Newton**. The bundled SO-101 tabletop scene opens above the node canvas with stopped physics, generated ground disabled, and the reference grid hidden. Drag the divider to resize it, or use the View menu to float and attach it inside the Blacknode window.
3. In the Newton app bar choose **File → Open scene…**. Blacknode's in-editor file browser accepts `.usd`, `.usda`, `.usdc`, `.urdf`, `.xacro`, `.xml`, and `.mjcf`; opening a file replaces the workspace stage and leaves simulation stopped at the initial state. URDF is loaded through Newton's native importer. Xacro is expanded with its standard processor before following the same URDF path, including relative mesh and texture resolution. MuJoCo XML/MJCF uses Newton's native MJCF importer, including relative `<include>` files and mesh directories. The imported Newton model is exported to a temporary USD render layer shared by Viser and OVRTX; the source files remain unchanged. Solid debug colliders are authored hidden and remain available through **View → Collision geometry** as a diagnostic overlay. An authored MJCF `<freejoint>` is preserved, a named `home` keyframe seeds the initial robot pose, and Blacknode adds a generated ground plane only when the selected MJCF include tree does not already contain a world plane. When a Xacro reads `$(env NAME)`, the editor asks for the missing value and retries expansion with a request-scoped override; it does not require restarting Blacknode or changing the editor server's global environment.

For MuJoCo Menagerie robots, select the model's `scene.xml` when available so its authored floor and scene settings are included. For Unitree Go2, open `unitree_go2/scene.xml`; `go2.xml` also loads and receives Blacknode's generated ground plane because that robot-only file has no world plane.
4. **View → Visual geometry** masks every rendered scene mesh. **View → Collision geometry** displays collision shapes as translucent per-shape colors. USD meshes used for both appearance and collision receive display-only collider proxies; these proxies do not add physics bodies or contacts.
4. Use **Window** to show the Maya-style **Outliner**, **Properties**, and **Robot Controller**. Disclosure arrows on the left expand and collapse hierarchy branches; visibility eyes stay aligned on the right. Generated ground is always listed so it can be hidden and restored; in OVRT it also exposes live transform and PreviewSurface material controls. The **Lights** branch exposes **Key / Sun** and **HDRI Environment** separately. Its parent eye, or **View → All lights**, hides or restores both together. Select the distant light to edit its enabled state, intensity, color, angular softness, and XYZ direction live. Color pickers display sRGB swatches and convert material and light values to linear RGB at the USD renderer boundary. Select an Outliner object or click rendered geometry to edit its local transform, visibility, and preview-surface material. Grid, object, light, material, environment, and joint-drive changes are live and keep the renderer, camera, and simulation session active. Persisted edits are authored into a temporary USD override layer; the source asset stays unchanged.
5. Use **View → Grid** to hide or show the reference grid. **View → Visual geometry** and **View → Collision geometry** independently switch the rendered model and color-coded collider shapes while the viewer and simulation keep running. Collider visibility is only a diagnostic overlay and does not enable, disable, or otherwise change collision physics. **View → Environment** controls the background, bundled HDRI preset, lighting intensity, and custom `.hdr`/`.exr` selection as values change. OVRT renders custom HDRI files; Viser renders its bundled presets.
6. Use **Simulation → Start**, **Stop**, and **Reset**, or the matching transport buttons. Stop pauses physics while keeping the stage and viewer open. **File → New stage** returns to the empty stage and **Close Newton** shuts down the workspace service.
7. Explicitly arm the **Robot Controller** before moving joint sliders. Each joint shows its command, rate-limited target, measured position, child-link mass and inertia, and the active physics rate. Drive stiffness and damping stay in their displayed revolute or prismatic SI units and accept decimal or scientific notation, such as `100000000` or `1e8`. Target speed and maximum step use stable displayed units. These four fields commit on Enter or focus loss and retain bounded command-rate safety.

## Run the SO-101 grain-spill demo

Open the **SO-101 Grain Spill Demo** template and run the graph. It loads the same bundled tabletop scene, fills the green container with a procedural `18 × 18 × 10` grid of 3,240 Newton XPBD grains, selects CUDA and the Viser particle renderer, uses the Apartment HDRI at `1.0` intensity, hides the reference grid, and keeps generated ground disabled. The table and authored environment supply collision support.

The template replaces the container's dense mesh collider with five invisible box colliders attached to the same movable rigid body. The visible green container remains unchanged, while the simpler interior keeps the grains contained and avoids showing or solving dozens of generated convex hulls. Once the viewer opens, explicitly arm the **Robot Controller** and use the joint controls to lift or tip the container. Simulation motion remains disarmed until that operator action.

`NewtonUSDScene.particle_fill` accepts a procedural origin, XYZ dimensions and spacing, particle radius and mass, jitter, friction, cohesion, adhesion, a velocity limit, an RGB display color, and an optional compound container proxy. A fill is capped at 100,000 particles and rejects initial spacing that would overlap after jitter. Viser keeps one float32 particle buffer alive, updates only its positions, and applies gradient point shading so dense grain motion does not flash from per-frame handle replacement. The demo uses a 24 Hz simulation with eight collision substeps and four XPBD solver iterations, and publishes viewer updates at 20 Hz. The session reports the resolved `particle_count` and normalized fill settings.

### Try the OVRT renderer

The `viewer-ovrtx` component is separate and disabled by default because its pinned OVRT/OVStage release train is a large optional RTX dependency. In **Packages**, enable `viewer-ovrtx` for `blacknode-newton` and install that component's prerequisites. The Newton View menu then exposes **Open with OVRT (RTX)** and **Use OVRT (RTX) renderer**. Switching renderer restarts the current generic scene; it does not create a renderer-specific graph or scene format.

OVRT runs in a child process while Blacknode hosts the responsive embedded page. Newton body poses and editor changes are written into OVStage at monotonically increasing ordinals, OVRT reads `LdrColor`, and the page receives a loopback HTTP MJPEG stream. The composed render stage preserves the selected USD's authored units and Y-up or Z-up axis, and its initial camera frames authored render bounds. The bundled tabletop scene opens with its authored geometry framed and the reference grid hidden. Plain click performs renderer picking, synchronizes the Outliner selection, and marks the selected geometry with a subtle warm-yellow outline without a viewport name badge or pivot dot. Left drag orbits, middle drag pans, and right drag or the wheel zooms; the corresponding Alt-modified controls remain available, and double-click restores the initial camera. The viewport toolbox uses `Q` for Select, `W` for Move, `E` for Rotate, and `R` for Scale. Move arrows, object-space rotation rings, and scale handles are native USD geometry rendered in the same OVRT frame as the camera. Invisible browser hit targets send lightweight OVStage previews while dragging, then commit one authoritative workspace/Newton transform on release. Native browser image dragging is disabled and pointer motion is coalesced once per display frame. During HDRI camera navigation the panorama preview uses an interaction-resolution projection, then restores full viewport resolution shortly after the pointer stops. The process starts only after the browser proxy is available, so the viewer can show native renderer initialization and shader compilation progress. A newly encountered RTX material/shader set can make the first usable frame slow; initialization publishes the first valid `LdrColor` frame immediately.

The current OVRT surface is an experimental renderer qualification. Simulation start/stop/reset, scene editing, materials, and articulation control live in the Newton app shell for both providers. OVRT additionally accepts local `.hdr` and `.exr` dome-light textures and sends them directly through RTX for image-based lighting, reflections, and a full-HD viewport background. OVRT requires a supported NVIDIA RTX GPU and driver; its component reports unavailable independently when the optional native packages are absent.

The workspace is a managed Newton service and does not create or modify graph nodes. `NewtonUSDScene`, `NewtonViewerConfig`, `NewtonSimulation`, and `NewtonJointCommand` remain the typed workflow interface when a graph needs to create a separately configured scene, control lifecycle, or automate joint targets. The editor workspace and workflow sessions use distinct run identifiers.

Visual-only USD files are accepted and displayed. Their status reports zero authored dynamic bodies and a warning that collision geometry is absent. Physics motion requires authored rigid bodies/colliders or explicit bodies supplied by a workflow scene.

Open the viewer's **View** folder to change the solid background, select a Viser HDRI preset and its intensity, show or hide the HDRI background, truly suppress Newton's physics grid, set camera position and orbit target, select the world up axis, and change keyboard translation speed. The Viser provider keeps unchanged plane handles alive across frames, so a stationary physics grid is not removed and recreated by the renderer. **Apply camera** uses the entered values; **Frame scene** restores an automatic view. Left-drag orbits around the displayed target, right-drag pans, and scrolling zooms.

Viser 1.x exposes its bundled HDRI presets (`apartment`, `city`, `dawn`, `forest`, `lobby`, `night`, `park`, `studio`, `sunset`, and `warehouse`) through the viewer API. OVRT resolves those same bundled panoramas into live dome-light textures and also accepts custom `.hdr` and `.exr` files. OVRT swaps a dome-only stage reference when the selected file changes, preserving the HDRI's authored asset type while the renderer, scene, camera, and simulation remain active. `hdri=none` uses the selected solid background. When an HDRI is active, the background-color picker changes only the fallback/hidden background and does not tint image-based lighting. **Show HDRI background** controls only camera visibility: disabling it keeps image-based lighting and reflections active. Intensity changes the active dome lighting in real time. The scene's distant key/sun remains enabled independently, including when dome intensity is `0`.

## Perception views

The OVRT viewer exposes five live display modes in both the viewport toolbar and the Newton **Perception** panel:

- **RGB** displays the tone-mapped RTX image.
- **Depth IR** maps metric camera distance to an infrared heat palette, with nearby surfaces shown hotter.
- **Segments** assigns a stable high-contrast color to every visible USD geometry instance.
- **Boxes** draws tight colored 2D bounds and object labels over RGB.
- **Composite** blends the segmentation colors with RGB and adds labeled bounds.

Blacknode authors semantic class and object-label metadata into the temporary workspace override layer. The source USD remains unchanged. Detection boxes are computed from visible semantic pixels, so they reflect occlusion and the current camera rather than loose projected world bounds. These modes provide renderer ground truth for scene inspection and synthetic-data work; inference over real camera images remains a perception-provider workflow.

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

## Mirror a monitored USB robot

Open a calibrated robot in **Robot Monitor**, open the matching articulation in the Newton workspace, and press **Drive Newton robot** in the monitor header. Blacknode reuses the monitor's shared normalized telemetry stream, matches joints by name, converts calibrated degree feedback to Newton's SI units, hides the comparison ghost, and drives the visible fixed-base articulation. The monitor and Newton connection share the existing hardware reader instead of opening a competing USB session.

Press **Calibration home** to resume and arm Newton and move only the simulated articulation to the exact per-joint `home_offset_deg` values saved by the active hardware calibration. This leaves the scene and its objects in place. Values are converted to Newton's SI units and clamped to the articulation's USD joint limits, so a saved hardware home outside the model's permitted range is reported as clamped.

Pressing **Drive Newton robot** is the explicit simulation-only authorization: it resumes and arms Newton, hides the reference ghost, and keeps the visible articulation advancing while that fresh monitor stream owns the follower. The monitor publishes at 20 Hz, coalesces pending frames to the newest pose, and receives compact control acknowledgements instead of full workspace snapshots. The default editor workspace tracks at up to 180 degrees per second with a bounded six-degree step. It never arms or commands the physical robot. Robot Monitor positions are already relative to the saved physical `home_ticks`, so every target is computed as the calibration's measured home offset plus that calibrated displacement. The USD-authored home remains the fallback for calibration records that do not contain an offset for a matching joint. Targets are recalculated from home on every frame and never accumulate against the previous frame. If telemetry becomes stale, disconnects, or the monitor closes, the stream watchdog disarms Newton. **Stop Newton follow** stops and disarms the follower. Pausing the workspace also revokes the follower, so motion does not restart until **Drive Newton robot** is pressed again. Raw tick telemetry and samples without an active calibration are rejected.

The bundled SO-101 tabletop scene replaces only the two inaccurate concave jaw physics meshes with 43 × 12 mm physical fingertip faces aligned along the full visible opposing grip surfaces, including the forward tips. Every table, mat, cube, and container retains active collision. Each movable container uses five physical box shapes for its floor and walls, preserving the open cavity while avoiding runtime mesh decomposition. The cubes and containers use a consistent 30-gram simulation mass. Newton resolves every grasp through fingertip, object, friction, and rigid-body contacts. The gripper uses a compliant XPBD position drive: its requested target continues closing while contact constraints physically stop the measured joint. The opposing fingertip pair also has self-contact enabled so empty jaws cannot cross. No object pose, velocity, or collision flag changes during a grasp. The editor workspace runs eight substeps and 24 solver iterations so thin finger contacts remain stable during closing and lift motions. Other opened assets keep their authored or configured contact values.

The default SO-101 profile and bundled SO-101 USD use the same six semantic joint names, so they connect automatically. Other clients can supply the Newton runtime's incoming `joint_map`, with monitor names as keys and Newton articulation names as values.

## Windows and ROS 2

The viewer works directly on Windows: Viser serves the page and streams the scene over WebSocket from the Blacknode process. ROS 2 is a separate transport boundary.

To connect a ROS graph from Windows:

1. Run `rosbridge_websocket` on a machine or container with ROS 2, reachable on TCP port 9090.
2. Enable `blacknode-ros2/rosbridge`, then enable the optional `blacknode-newton/rosbridge` component in **Packages**.
3. Start the Newton simulation, connect `NewtonSimulation.session` to `NewtonROSBridge.session`, then start the bridge with the rosbridge host.
4. Read `/blacknode/newton/joint_states` and publish named SI-unit targets to `/blacknode/newton/joint_commands` as `sensor_msgs/msg/JointState` (radians for revolute joints and metres for prismatic joints).

For real robot → Newton mirroring, set `direction=ros_to_newton` and set `command_topic` to the real robot's `JointState` topic (commonly `/joint_states`). Use `joint_map` when ROS and USD joint names differ; its keys are incoming ROS names and its values are Newton articulation names. Extra ROS joints are ignored. Incoming joint telemetry appears immediately in the Robot Controller's **Digital Twin** card, including live/stale state, message age, source latency, matched joints, and simulation tracking error. Each mapped joint shows the real/reference position and signed Newton delta. A bounded live trace plots maximum and RMS tracking error over time so lag, jitter, and overshoot remain visible; **Clear trace** resets that diagnostic history without restarting the viewer, simulation, or transport.

The current real/reference pose is also rendered as a translucent cyan robot ghost. It can sit beside Newton for an uncluttered comparison, overlay the simulated articulation for alignment inspection, use a custom XYZ offset, or be hidden live. The ghost is always read-only. **Sync Newton once to real pose** copies one fresh reference pose into simulation through the existing explicit arm, joint-limit, command-rate, and stale-data gates; it never commands the physical robot. Arm the Newton simulation explicitly to let the event-driven subscription update corresponding Newton targets; the bridge disarms it if feedback becomes stale.

Use **Saved tracking runs → Save trace** to persist the current bounded trace as a `blacknode.newton-run-artifact`. The artifact records source and receive timestamps, signed per-joint errors, joint names and units, scene identity, physics configuration, and aggregate maximum/RMS error. Files and their atomic index are stored under Blacknode's local `newton-runs` data directory and registered as `simulation_run` evidence in the Blacknode artifact catalog. Select a prior run and press **Compare** to overlay its maximum-error curve as a read-only baseline. Only joints with matching names and units participate; loading or clearing a baseline never changes simulation state or motion authorization.

For Newton → real robot mirroring, set `direction=newton_to_ros` and publish Newton state on a dedicated topic such as `/blacknode/newton/joint_states`. Connect a live `ROS2SubscribeJointState.subscription` to `ROS2JointController.subscription`, connect the calibrated `Robot.robot` descriptor to `ROS2JointController.robot`, and set `require_leader_released=false` because Newton is the source rather than a torque-released physical leader. Keep the controller disarmed while verifying joint mapping, limits, current feedback, and direction. Arm that controller only after its status is fresh and the robot is in a safe operating area. The controller synchronizes from the physical robot's current pose, enforces calibration and joint limits, expires stale input, and owns torque release on stop.

Use different ROS topics for Newton state and Newton command input. The bridge rejects a bidirectional configuration that uses the same topic for both legs. Avoid running real → Newton and Newton → real simultaneously for the same robot unless an explicit motion arbiter owns the loop; otherwise feedback can become a command cycle.

The simulator still starts disarmed. ROS motion commands are rejected until the corresponding Newton session is explicitly armed, while their joint observations continue feeding the read-only Digital Twin comparison. After a ROS command is accepted, the bridge disarms the session if that command stream is stale for more than `command_stale_seconds` (default `0.5 s`).

The Viser server listens on all interfaces so another machine can use the host's LAN address instead of `127.0.0.1`. Treat the port as an operator control endpoint: expose it only on a trusted network or behind an authenticated reverse proxy.

## Replay a dataset episode on the loaded robot

Enable `blacknode-dataset/publishing` and the optional `blacknode-newton/replay` component, open a robot in the Newton workspace, then open the **Replay Dataset Episode in Loaded Newton Robot** template. Set the Dataset Browser root, dataset, episode, and camera. Running the graph starts a loopback `StreamPublisher`, resumes and explicitly arms the loaded simulation, and starts `NewtonReplayBridge`. Play and timeline seeks in Dataset Browser then update the matching Newton articulation joints in real time.

The publisher's `source` selects recorded `action`, `observation`, or `leader` joint values. Names match directly by default. If the dataset and USD articulation use different names, set `NewtonReplayBridge.joint_map` with recorded names as keys and Newton names as values, for example `{"shoulder": "shoulder_lift"}`. Extra recorded joints are ignored. `units=auto` reads the replay frame units and converts degrees to radians for revolute Newton joints. Replay frames feed the same Digital Twin metrics and tracking-history chart as live ROS telemetry, including while Newton motion is disarmed.

Dataset replay and ROS 2 reach the same managed Newton command session, including explicit arming, joint limits, rate limiting, and stale-command disarm. Their transports are intentionally separate: dataset replay is a deterministic recorded timeline over the Dataset `StreamPublisher` WebSocket, while ROS 2 is a live topic connection that can also publish Newton state. The replay bridge commands simulation only. Driving a physical robot from a dataset remains a separately armed robot-controller workflow with calibration, feedback, limits, stale-data handling, and an operator-owned motion boundary.

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
