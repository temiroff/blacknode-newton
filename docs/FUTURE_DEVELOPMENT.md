# Future development: Newton simulation platform

This document records planned features. It is intentionally separate from the package manifest: none of these capabilities are declared as available until implementation and acceptance tests exist.

## Product direction

Grow the generic USD simulation surface into repeatable tests, dataset capture, policy evaluation, and reinforcement learning. Stable Blacknode scene, robot, observation, action, and run-artifact contracts remain above replaceable physics, viewer, and transport providers.

## Implemented renderer baseline

The optional `viewer-ovrtx` component now qualifies the provider boundary with real USD rendering, Newton body-pose synchronization through OVStage, an embedded RTX stream, and orbit/pan/zoom camera input. It remains separate from the default Viser component and uses pinned OVRT 0.4/OVStage 0.1 packages while those APIs are pre-release.

## 1. Viewer providers, including OVUI

Add an `ovui` provider beside `viser`; do not change `NewtonSession` or saved workflow ports. Qualification requires:

- USD geometry, materials, articulation motion, contacts, and camera navigation.
- The same arm/disarm and joint-target operations as the Viser provider.
- Clean provider startup/shutdown and structured unavailable status when OVUI is absent.
- Side-by-side contract tests that execute identical scene frames through both providers.

OVUI should be evaluated for editor embedding, richer scene inspection, gizmos, selection, and high-fidelity NVIDIA rendering. Viser remains useful for lightweight browser/WebSocket access on Windows and remote machines.

OVUI evaluation should reuse the current provider contract and OVRT renderer rather than coupling application chrome to NewtonSession. Qualification must determine whether OVUI can replace the current small HTTP surface while preserving editor attach/float behavior and headless operation.

## 2. General scene and robot import

- Add a URDF importer that emits the same generic scene contract as the implemented USD importer.
- Persist stable joint/link names, units, limits, collision status, and sensor frames.
- Add scene composition for tables, bins, lights, cameras, and multiple rigid objects.
- Add per-asset import qualification that fails when an articulation has missing collision geometry, unbounded joints, invalid mass/inertia, or unsupported schemas.

## 3. Transport providers

- Native `rclpy` adapter for Linux/ROS hosts.
- Current rosbridge adapter for Windows and remote ROS graphs.
- Blacknode direct WebSocket transport for low-latency editor controls and observations.
- Contract tests across native ROS, rosbridge, replay, and in-process transports, including stale-command disarming and reconnect behavior.

Viewer streaming and ROS transport remain independent: a browser can view a native ROS-connected run, and a ROS client can control a headless run.

## 4. Sensors and recording

- Newton camera, depth, segmentation, contact, joint-state, and force/torque observations.
- A provider-neutral viewer-overlay contract for trajectories, line strips, contact normals, coordinate frames, point clouds, and labels; Viser and future viewers implement the same overlay IDs and updates.
- Semantic and instance-segmentation render outputs with stable object IDs. Treat these as synchronized sensor observations rather than colors sampled from the operator viewer.
- Time-synchronized `blacknode.sim-observation` records with scene revision, action, seed, and physics configuration.
- Chunked run artifacts for deterministic replay, inspection in the editor, and conversion to Blacknode datasets.
- Replay provider that implements the same robot/sensor contracts with no simulator or hardware dependency.

## 5. Scenario tests

Introduce declarative scenario nodes for reset conditions, disturbances, assertions, and metrics. Initial qualifications:

- cube settles on the ground;
- SO-101 tracks a bounded joint target;
- gripper establishes two-sided cube contact;
- cube rises by at least 50 mm and stays held for a specified interval;
- reset reproduces initial state within tolerance;
- seeded runs reproduce metric traces.

These become CI acceptance tests and editor-visible reports rather than ad hoc scripts.

## 6. Policy test and replay

- Consume Blacknode policy artifacts through an observation/action adapter owned by `blacknode-motion` contracts.
- Execute a policy with explicit arm, pause, takeover, and emergency-stop state even in simulation.
- Record observations, raw predictions, clamped actions, rewards, contacts, and policy/model revisions.
- Replay either recorded actions or policy inference against recorded observations; highlight the first divergence.
- Run batch evaluation over seeds and scene variations, producing success-rate and safety summaries.

## 7. Reinforcement learning

- Vectorized, headless Newton environments separated from the interactive viewer loop.
- Gymnasium-compatible environment adapter with observation/action specifications generated from Blacknode contracts.
- Task contracts for reset, reward, termination, curriculum, and domain randomization.
- Training adapters for an initial supported framework selected by measured Newton compatibility.
- Checkpoint export to `blacknode.policy-artifact`, followed by deterministic evaluation in the same scenario contract and an interactive viewer replay.
- Promotion gates based on task success, joint-limit violations, contact impulses, inference latency, and held-out randomized scenes.

## Delivery sequence

1. Add an automated SO-101 grasp/lift scenario qualification over the generic scene contract and persist a deterministic acceptance trajectory.
2. Add observation and run-artifact recording/replay.
3. Add URDF articulation import and richer USD scene composition.
4. Add native ROS 2 and direct Blacknode WebSocket transport providers.
5. Evaluate OVUI as an application-surface replacement above the implemented OVRT provider contract.
6. Add policy evaluation/replay, then vectorized RL training.

Each milestone must ship code, package dependencies, UI-visible availability, tests, and a portable template together.
