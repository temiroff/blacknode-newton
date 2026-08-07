# blacknode-newton

`blacknode-newton` adds managed Newton physics sessions to the Blacknode editor. It loads USD and common robot-description formats, renders the live scene in an embedded viewer, exposes articulation controls, and connects simulation to workflows, ROS 2, or recorded datasets.

## Components

| Component | Default | Purpose |
|---|---:|---|
| `runtime` | On | Scene loading, Newton XPBD simulation, articulation state, limits, and safe commands |
| `viewer-viser` | On | Embedded browser viewer and robot-control panel |
| `viewer-ovrtx` | Off | Optional RTX viewer using OVRT/OVStage |
| `rosbridge` | Off | ROS 2 `JointState` input and output over rosbridge |
| `replay` | Off | Dataset episode playback into the active Newton articulation |

The workflow nodes are `NewtonUSDScene`, `NewtonViewerConfig`,
`NewtonSimulation`, `NewtonJointCommand`, and `SO101ReachTask`. The reach task
defines a vectorized, simulation-only SO-ARM101 environment for reinforcement
learning. Optional transports add `NewtonROSBridge` and `NewtonReplayBridge`.

## Quick start

1. In **Packages**, enable `runtime` and `viewer-viser`, then press **Install prerequisites**.
2. Press **Open Newton**. The bundled SO-101 tabletop scene opens with physics stopped.
3. Use **File → Open scene…** for USD, URDF, Xacro, XML, or MuJoCo MJCF assets.
4. Use **View** and **Window** to control geometry, lights, environment, Outliner, Properties, and Robot Controller.
5. Start or reset simulation from **Simulation**. Explicitly arm the Robot Controller before moving joints.

Scene edits are written to a temporary override layer; source assets stay unchanged. Joint targets are clamped to authored limits and rate-limited before they reach Newton.

Enable `viewer-ovrtx` to expose the optional RTX viewer. It requires a supported NVIDIA RTX GPU and reports an independent unavailable state when its native dependencies are missing.

## Included workflows

| Template | Purpose |
|---|---|
| `usd-scene-viewer.json` | Load and inspect a scene with managed simulation and viewing |
| `so101-grain-spill-demo.json` | Run the bundled SO-101 tabletop particle demo |
| `ros2-newton-joint-sync.json` | Mirror named ROS 2 joint state into Newton |
| `dataset-episode-newton-replay.json` | Replay a recorded dataset episode on the loaded robot |

The `blacknode-training` package adds `so101-ppo-training.json`, which connects
`SO101ReachTask` to managed PPO training, simulation evaluation, and policy
artifact export. Its optional Viser preview renders one sampled training
environment with a target marker, end-effector trail, metrics, and environment
selector while the full batch continues stepping on the training device.

Robot Monitor can also drive a matching Newton articulation from fresh calibrated telemetry. This authorizes simulation only; it never commands the physical robot. Stale or disconnected streams disarm the Newton follower.

## Safety

- Sessions start stopped and disarmed.
- Joint limits, freshness checks, and bounded command rates remain active for editor, ROS, and replay inputs.
- Pause, stop, stale input, or transport loss revokes motion authorization.
- Self-collision is opt-in.
- ROS and dataset bridges command the managed simulation session only.
- Expose viewer and rosbridge ports only on trusted networks.

## Development

From this package worktree:

```powershell
$env:PYTHONPATH = "..\..\python"
python -m unittest discover -s tests -v
```

Architecture details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Future work is tracked in [docs/FUTURE_DEVELOPMENT.md](docs/FUTURE_DEVELOPMENT.md). The package is Apache-2.0; bundled model attribution is recorded in [NOTICE](NOTICE).
