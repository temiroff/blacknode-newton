# Current architecture

The first release is one complete simulation slice with three replaceable boundaries.

```text
Blacknode workflow
  NewtonUSDScene ─────── scene contract ──┐
  NewtonViewerConfig ─ viewer contract ───┤
                                          v
                                  NewtonSession / XPBD
                                    |       |       |
                           viewer contract  |  state snapshots
                              /       \     |       |
                         Viser       OVRT/OVStage   optional ROS bridge
                           |            |
                     browser WS   isolated RTX worker
                                      |
                               embedded MJPEG proxy

                                  safe joint command API
```

## Scene contract

`blacknode.newton-scene` resolves a USD, URDF/Xacro, or MuJoCo MJCF asset, base mode, optional home values, ground configuration, optional rigid bodies, one bounded procedural particle fill, and collision overrides. Native Newton importers build non-USD robot descriptions, then export frame zero to a temporary USD render layer shared by every viewer provider. The runtime inspects revolute and prismatic joints from the imported model and rejects duplicate names or non-finite limits before enabling teleoperation.

Particle fills are expanded into Newton's GPU-friendly grid representation while building the managed session. The contract stores dimensions, physical material settings, display color, and an optional compound container collision proxy instead of persisting thousands of particle records. Validation caps the fill at 100,000 particles and prevents initial overlap. A compound proxy can replace one expensive mesh collider with bounded, invisible boxes attached to the same imported rigid body. XPBD's particle hash grid supplies particle-particle contact, while the imported scene and any compound proxy supply particle-shape contact. Viser maintains one float32 particle buffer with gradient shading and updates only its positions at the viewer's bounded render cadence.

During import, the runtime applies collision APIs to collision-authored meshes on an in-memory USD stage. Closed convex hulls are the default. Paths selected through `convex_decomposition_patterns` use bounded CoACD decomposition when concavity must be retained. Self-collision is an explicit scene option and remains off by default.

## Viewer contract

A viewer provider registers a factory name and exposes `url`, `is_running`, `begin_frame`, `log_state`, `end_frame`, and `close`. The physics runtime has no concrete renderer imports. The default `viewer-viser` and optional `viewer-ovrtx` components register independently, and the editor embeds either reported URL in the same resizable viewer-above-canvas split.

The OVRT provider isolates native renderer initialization and shutdown in a child process. Renderer creation follows NVIDIA's single-threaded application flow; only after creation does the worker start consuming pose messages. The worker composes the selected asset and Blacknode's camera, lights, render product, ground, and grid into one inline root layer through OVStage. Source `metersPerUnit`, Y-up or Z-up orientation, and authored render bounds drive stage metadata, pose conversion, and initial camera framing. OVStage receives batched `omni:xform` matrices with a new ordinal and advanced write floor for each update. Render mappings are copied before unmapping, and queries, path lists, render products, OVStage, and OVRT are released in dependency order. A lightweight HTTP proxy remains responsive during shader compilation and streams completed `LdrColor` frames to the editor.

## Command contract

All viewer, workflow, and ROS targets pass through the same named-joint API. A session starts disarmed, validates finite values, rejects unknown joints, clamps USD limits, and rate-limits the applied target. Arming seeds the target from the observed articulation pose to prevent a jump. The ROS bridge adds a stale-command watchdog and disarms when an accepted ROS stream times out.

## Managed lifecycle

Sessions are keyed by `run_id`. Repeated starts with the same contract reconcile to the running session; conflicting starts fail. Stop requests close the loop, viewer, and registered transport hooks, and an `atexit` hook disarms and stops remaining services. Blacknode's editor runtime registry includes the package so **Stop all** reports and closes its sessions.
