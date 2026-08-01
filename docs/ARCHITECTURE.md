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

`blacknode.newton-scene` resolves a USD asset, root prim, base mode, optional home values, ground configuration, optional rigid bodies, and collision overrides. The runtime inspects revolute and prismatic joints directly from the imported USD and rejects duplicate names or non-finite limits before enabling teleoperation.

During import, the runtime applies collision APIs to collision-authored meshes on an in-memory USD stage. Closed convex hulls are the default. Paths selected through `convex_decomposition_patterns` use bounded CoACD decomposition when concavity must be retained. Self-collision is an explicit scene option and remains off by default.

## Viewer contract

A viewer provider registers a factory name and exposes `url`, `is_running`, `begin_frame`, `log_state`, `end_frame`, and `close`. The physics runtime has no concrete renderer imports. The default `viewer-viser` and optional `viewer-ovrtx` components register independently, and the editor embeds either reported URL in the same resizable viewer-above-canvas split.

The OVRT provider isolates native renderer initialization and shutdown in a child process. Renderer creation follows NVIDIA's single-threaded application flow; only after creation does the worker start consuming pose messages. The worker composes the selected asset and Blacknode's camera, lights, render product, ground, and grid into one inline root layer through OVStage. Source `metersPerUnit`, Y-up or Z-up orientation, and authored render bounds drive stage metadata, pose conversion, and initial camera framing. OVStage receives batched `omni:xform` matrices with a new ordinal and advanced write floor for each update. Render mappings are copied before unmapping, and queries, path lists, render products, OVStage, and OVRT are released in dependency order. A lightweight HTTP proxy remains responsive during shader compilation and streams completed `LdrColor` frames to the editor.

## Command contract

All viewer, workflow, and ROS targets pass through the same named-joint API. A session starts disarmed, validates finite values, rejects unknown joints, clamps USD limits, and rate-limits the applied target. Arming seeds the target from the observed articulation pose to prevent a jump. The ROS bridge adds a stale-command watchdog and disarms when an accepted ROS stream times out.

## Managed lifecycle

Sessions are keyed by `run_id`. Repeated starts with the same contract reconcile to the running session; conflicting starts fail. Stop requests close the loop, viewer, and registered transport hooks, and an `atexit` hook disarms and stops remaining services. Blacknode's editor runtime registry includes the package so **Stop all** reports and closes its sessions.
