"""Vectorized Newton/Warp reinforcement-learning environments for SO-ARM101."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Dict, Float, Int, List, Text, node


_CATEGORY = "Newton Simulation"
_ASSET_URI = "package://blacknode-newton/assets/so101_robot.usd"
SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
OBSERVATION_DIM = 21
ACTION_DIM = len(SO101_JOINT_NAMES)


def _asset_path(value: str) -> Path:
    requested = str(value or _ASSET_URI).strip()
    if requested == _ASSET_URI:
        return (Path(__file__).resolve().parents[1] / "assets" / "so101_robot.usd").resolve()
    path = Path(requested).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SO-ARM101 Newton asset does not exist: {path}")
    return path


def environment_spec(
    *,
    asset_path: str = _ASSET_URI,
    environment_count: int = 512,
    episode_steps: int = 128,
    simulation_hz: int = 120,
    control_hz: int = 30,
    action_scale_deg: float = 3.0,
    success_tolerance_m: float = 0.025,
    seed: int = 42,
) -> dict[str, Any]:
    simulation_hz = max(30, min(1000, int(simulation_hz)))
    control_hz = max(10, min(240, int(control_hz)))
    if simulation_hz < control_hz or simulation_hz % control_hz:
        raise ValueError("simulation_hz must be an integer multiple of control_hz")
    return {
        "kind": "blacknode.rl-environment",
        "schema_version": 1,
        "provider": {
            "package": "blacknode-newton",
            "component": "runtime",
            "environment_type": "so101-reach-v1",
        },
        "task": "reach",
        "robot_profile": "so_arm101",
        "asset_path": str(asset_path or _ASSET_URI),
        "environment_count": max(1, min(8192, int(environment_count))),
        "episode_steps": max(8, min(4096, int(episode_steps))),
        "simulation_hz": simulation_hz,
        "control_hz": control_hz,
        "action_scale_rad": math.radians(max(0.05, min(20.0, float(action_scale_deg)))),
        "success_tolerance_m": max(0.005, min(0.2, float(success_tolerance_m))),
        "seed": int(seed),
        "joint_names": list(SO101_JOINT_NAMES),
        "observation": {
            "dimension": OBSERVATION_DIM,
            "fields": [
                "normalized_joint_positions[6]",
                "joint_velocities[6]",
                "target_minus_end_effector_xyz[3]",
                "previous_action[6]",
            ],
        },
        "action": {
            "dimension": ACTION_DIM,
            "type": "bounded-joint-position-delta",
            "minimum": -1.0,
            "maximum": 1.0,
            "scale_rad": math.radians(max(0.05, min(20.0, float(action_scale_deg)))),
            "units": "normalized",
        },
        "reward": {
            "progress": 12.0,
            "distance": -2.0,
            "success": 10.0,
            "action": -0.01,
            "velocity": -0.0005,
            "joint_limit": -0.05,
        },
        "safety": {
            "simulation_only": True,
            "physical_motion_authorized": False,
            "joint_limits_from_asset": True,
            "self_collisions": False,
        },
    }


def _resolve_compute_device(torch: Any, wp: Any, device: str) -> str:
    requested = str(device or "auto").strip().lower()
    warp_cuda_available = bool(wp.is_cuda_available())
    torch_cuda_available = bool(
        getattr(torch.version, "cuda", None) and torch.cuda.is_available()
    )
    if requested == "auto":
        requested = "cuda:0" if warp_cuda_available and torch_cuda_available else "cpu"
    elif requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not warp_cuda_available:
        raise RuntimeError("CUDA was requested but Warp cannot access a CUDA device")
    if requested.startswith("cuda") and not torch_cuda_available:
        raise RuntimeError(
            "CUDA was requested but this Python environment has a CPU-only PyTorch build. "
            "Install a CUDA-enabled PyTorch build in the Blacknode editor environment, "
            "then restart the Python server."
        )
    return str(wp.get_device(requested))


class SO101ReachEnvironment:
    """Batched SO-ARM101 reach task with Newton dynamics and Torch views.

    The environment owns no physical transport. Actions only update the
    controls of replicated simulated articulations.
    """

    def __init__(self, spec: dict[str, Any], device: str = "auto") -> None:
        if spec.get("kind") != "blacknode.rl-environment":
            raise ValueError("connect a blacknode.rl-environment")
        provider = spec.get("provider") if isinstance(spec.get("provider"), dict) else {}
        if provider.get("environment_type") != "so101-reach-v1":
            raise ValueError("SO101ReachEnvironment requires environment_type so101-reach-v1")
        try:
            import newton
            import torch
            import warp as wp
            from newton.selection import ArticulationView
        except Exception as exc:  # pragma: no cover - package diagnostics own dependency setup
            raise RuntimeError("Newton, Warp, and PyTorch are required for SO-ARM101 RL") from exc

        self.newton = newton
        self.torch = torch
        self.wp = wp
        self.spec = dict(spec)
        self.device = _resolve_compute_device(torch, wp, device)
        self.torch_device = torch.device(self.device)
        self.environment_count = int(spec.get("environment_count") or 512)
        self.episode_steps = int(spec.get("episode_steps") or 128)
        self.simulation_hz = int(spec.get("simulation_hz") or 120)
        self.control_hz = int(spec.get("control_hz") or 30)
        if self.simulation_hz < self.control_hz or self.simulation_hz % self.control_hz:
            raise ValueError("simulation_hz must be an integer multiple of control_hz")
        self.frame_skip = self.simulation_hz // self.control_hz
        self.simulation_dt = 1.0 / float(self.simulation_hz)
        self.action_scale_rad = float(spec.get("action_scale_rad") or math.radians(3.0))
        self.success_tolerance_m = float(spec.get("success_tolerance_m") or 0.025)
        self.observation_dim = OBSERVATION_DIM
        self.action_dim = ACTION_DIM
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(int(spec.get("seed") or 42))
        self.preview_environment_index = 0
        self.preview_viewer = None
        self.preview_model = None
        self.preview_state = None
        self.preview_frame = 0
        self.preview_error = ""

        template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        template.add_usd(
            str(_asset_path(str(spec.get("asset_path") or _ASSET_URI))),
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        self.template = template
        joint_entries: list[tuple[str, int, int]] = []
        for joint_id, label in enumerate(template.joint_label):
            name = str(label).rsplit("/", 1)[-1]
            if name in SO101_JOINT_NAMES:
                joint_entries.append((name, int(template.joint_q_start[joint_id]), int(template.joint_qd_start[joint_id])))
        if [entry[0] for entry in joint_entries] != list(SO101_JOINT_NAMES):
            raise RuntimeError("SO-ARM101 USD joint order does not match the robot profile contract")
        self._lower_values = [float(template.joint_limit_lower[drive]) for _, _, drive in joint_entries]
        self._upper_values = [float(template.joint_limit_upper[drive]) for _, _, drive in joint_entries]
        if not all(math.isfinite(lo) and math.isfinite(hi) and lo < hi for lo, hi in zip(self._lower_values, self._upper_values)):
            raise RuntimeError("SO-ARM101 USD requires finite ordered joint limits")

        scene = newton.ModelBuilder()
        scene.replicate(template, world_count=self.environment_count)
        self.model = scene.finalize(device=self.device)
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)
        self.view = ArticulationView(self.model, "/so101_new_calib", verbose=False)
        if int(self.view.count) != self.environment_count:
            raise RuntimeError("Newton did not replicate the expected SO-ARM101 articulations")
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.goal_state = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.bodies_per_environment = self.model.body_count // self.environment_count
        body_names = [str(value).rsplit("/", 1)[-1] for value in template.body_label]
        try:
            self.end_effector_body = body_names.index("gripper_frame_link")
        except ValueError as exc:
            raise RuntimeError("SO-ARM101 USD is missing gripper_frame_link") from exc

        self.lower = torch.tensor(self._lower_values, device=self.torch_device, dtype=torch.float32)
        self.upper = torch.tensor(self._upper_values, device=self.torch_device, dtype=torch.float32)
        self.mid = (self.lower + self.upper) * 0.5
        self.half_range = (self.upper - self.lower) * 0.5
        self.previous_actions = torch.zeros(
            (self.environment_count, self.action_dim), device=self.torch_device, dtype=torch.float32
        )
        self.targets = torch.zeros((self.environment_count, 3), device=self.torch_device, dtype=torch.float32)
        self.step_counts = torch.zeros(self.environment_count, device=self.torch_device, dtype=torch.int64)
        self.previous_distance = torch.zeros(self.environment_count, device=self.torch_device, dtype=torch.float32)
        self.total_resets = 0
        self.reset()

    def set_preview_environment(self, environment_index: int) -> int:
        self.preview_environment_index = max(
            0, min(self.environment_count - 1, int(environment_index))
        )
        return self.preview_environment_index

    def start_preview(self, config: dict[str, Any]) -> dict[str, Any]:
        """Start a read-only viewer backed by a separate one-arm model."""
        from .viewer_contract import create_viewer

        if self.preview_viewer is not None and self.preview_viewer.is_running():
            return self.preview_status()
        self.set_preview_environment(int(config.get("environment_index") or 0))
        self.preview_model = self.template.finalize(device=self.device)
        self.preview_state = self.preview_model.state()
        self.newton.eval_fk(
            self.preview_model, self.preview_model.joint_q, self.preview_model.joint_qd,
            self.preview_state,
        )
        viewer_config = {
            "mode": "training-preview",
            "port": max(1024, min(65535, int(config.get("port") or 8091))),
            "label": str(config.get("label") or "SO-ARM101 PPO Training"),
            "background_color": str(config.get("background_color") or "#111827"),
            "show_visuals": True,
        }
        self.preview_viewer = create_viewer(
            str(config.get("provider") or "viser"), self, self.preview_model,
            viewer_config,
        )
        self.preview_error = ""
        self.render_preview({"update": 0, "updates": 0, "reward": 0.0})
        return self.preview_status()

    def preview_status(self) -> dict[str, Any]:
        running = bool(
            self.preview_viewer is not None and self.preview_viewer.is_running()
        )
        return {
            "kind": "blacknode.rl-training-preview", "schema_version": 1,
            "running": running,
            "viewer_url": str(getattr(self.preview_viewer, "url", "") or ""),
            "environment_index": int(self.preview_environment_index),
            "environment_count": int(self.environment_count),
            "frame": int(self.preview_frame), "error": self.preview_error,
            "simulation_only": True, "physical_motion_authorized": False,
        }

    def render_preview(self, metadata: dict[str, Any] | None = None) -> bool:
        if self.preview_viewer is None or self.preview_state is None or self.preview_model is None:
            return False
        if not self.preview_viewer.is_running():
            return False
        index = self.set_preview_environment(self.preview_environment_index)
        source_q = self._attribute("joint_q", self.state_0)[index]
        source_qd = self._attribute("joint_qd", self.state_0)[index]
        preview_q = self.wp.to_torch(self.preview_state.joint_q).reshape(-1)
        preview_qd = self.wp.to_torch(self.preview_state.joint_qd).reshape(-1)
        preview_q[:] = source_q
        preview_qd[:] = source_qd
        self.newton.eval_fk(
            self.preview_model, self.preview_state.joint_q, self.preview_state.joint_qd,
            self.preview_state,
        )
        end_effector = self._end_effector(self.state_0)[index]
        distance = self.torch.linalg.vector_norm(self.targets[index] - end_effector)
        details = dict(metadata or {})
        details.update({
            "environment_index": index,
            "target_m": self.targets[index].detach().cpu().tolist(),
            "end_effector_m": end_effector.detach().cpu().tolist(),
            "distance_m": float(distance.detach().cpu()),
            "episode_step": int(self.step_counts[index].item()),
        })
        self.preview_viewer.begin_frame(self.preview_frame / max(1.0, float(self.control_hz)))
        log_training_state = getattr(self.preview_viewer, "log_training_state", None)
        if callable(log_training_state):
            log_training_state(self.preview_state, details)
        else:
            self.preview_viewer.log_state(self.preview_state)
        self.preview_viewer.end_frame()
        self.preview_frame += 1
        return True

    def _attribute(self, name: str, owner: Any) -> Any:
        return self.wp.to_torch(self.view.get_attribute(name, owner))[:, 0, :]

    def _end_effector(self, state: Any) -> Any:
        poses = self.wp.to_torch(state.body_q).reshape(
            self.environment_count, self.bodies_per_environment, 7
        )
        return poses[:, self.end_effector_body, :3]

    def _random_joint_positions(self, count: int, *, target: bool) -> Any:
        torch = self.torch
        margin = self.half_range * (0.18 if target else 0.72)
        low = self.lower + margin
        high = self.upper - margin
        values = low + torch.rand(
            (count, self.action_dim), generator=self.generator, device=self.torch_device
        ) * (high - low)
        if not target:
            values = self.mid + (values - self.mid) * 0.22
        # Reach does not train grasping yet. Hold the gripper near its midpoint
        # while preserving a six-joint action/artifact contract.
        values[:, -1] = self.mid[-1]
        return values

    def reset(self, mask: Any | None = None) -> Any:
        torch = self.torch
        if mask is None:
            mask = torch.ones(self.environment_count, device=self.torch_device, dtype=torch.bool)
        else:
            mask = mask.to(device=self.torch_device, dtype=torch.bool)
        count = int(mask.sum().item())
        if count == 0:
            return self.observe()
        world_mask = self.wp.from_torch(mask.contiguous())
        self.solver.reset(self.state_0, world_mask=world_mask)
        self.solver.reset(self.state_1, world_mask=world_mask)

        q = self._attribute("joint_q", self.state_0)
        qd = self._attribute("joint_qd", self.state_0)
        q[mask] = self._random_joint_positions(count, target=False)
        qd[mask] = 0.0
        self.newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        goal_q = self._attribute("joint_q", self.goal_state)
        goal_qd = self._attribute("joint_qd", self.goal_state)
        goal_q[mask] = self._random_joint_positions(count, target=True)
        goal_qd[mask] = 0.0
        self.newton.eval_fk(self.model, self.goal_state.joint_q, self.goal_state.joint_qd, self.goal_state)
        self.targets[mask] = self._end_effector(self.goal_state)[mask]

        control_targets = self._attribute("joint_target_q", self.control)
        control_targets[mask] = q[mask]
        self.previous_actions[mask] = 0.0
        self.step_counts[mask] = 0
        self.previous_distance[mask] = self.torch.linalg.vector_norm(
            self.targets[mask] - self._end_effector(self.state_0)[mask], dim=-1
        )
        self.total_resets += count
        return self.observe()

    def observe(self) -> Any:
        q = self._attribute("joint_q", self.state_0)
        qd = self._attribute("joint_qd", self.state_0)
        normalized_q = self.torch.clamp((q - self.mid) / self.half_range, -1.0, 1.0)
        delta = (self.targets - self._end_effector(self.state_0)) / 0.5
        return self.torch.cat((normalized_q, qd * 0.05, delta, self.previous_actions), dim=-1)

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        torch = self.torch
        actions = actions.to(device=self.torch_device, dtype=torch.float32)
        if tuple(actions.shape) != (self.environment_count, self.action_dim):
            raise ValueError(
                f"actions must have shape ({self.environment_count}, {self.action_dim}), got {tuple(actions.shape)}"
            )
        actions = torch.clamp(actions, -1.0, 1.0)
        q = self._attribute("joint_q", self.state_0)
        desired = torch.clamp(q + actions * self.action_scale_rad, self.lower, self.upper)
        # Keep the gripper fixed during the reach curriculum.
        desired[:, -1] = self.mid[-1]
        self._attribute("joint_target_q", self.control)[:] = desired
        for _ in range(self.frame_skip):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.simulation_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.step_counts += 1
        q = self._attribute("joint_q", self.state_0)
        qd = self._attribute("joint_qd", self.state_0)
        distance = torch.linalg.vector_norm(self.targets - self._end_effector(self.state_0), dim=-1)
        progress = self.previous_distance - distance
        normalized_q = torch.abs((q - self.mid) / self.half_range)
        limit_cost = torch.relu(normalized_q - 0.9).square().mean(dim=-1)
        success = distance <= self.success_tolerance_m
        finite = torch.isfinite(q).all(dim=-1) & torch.isfinite(qd).all(dim=-1) & torch.isfinite(distance)
        timeout = self.step_counts >= self.episode_steps
        done = success | timeout | ~finite
        reward = (
            12.0 * progress
            - 2.0 * distance
            - 0.01 * actions.square().mean(dim=-1)
            - 0.0005 * qd.square().mean(dim=-1)
            - 0.05 * limit_cost
            + 10.0 * success.to(torch.float32)
            - 20.0 * (~finite).to(torch.float32)
        )
        terminal_distance = distance.detach().clone()
        self.previous_distance = distance
        self.previous_actions = actions
        if bool(done.any()):
            self.reset(done)
        observation = self.observe()
        return observation, reward, done, {
            "distance_m": terminal_distance,
            "success": success,
            "timeout": timeout,
            "finite": finite,
        }

    def close(self) -> None:
        if self.preview_viewer is not None:
            try:
                self.preview_viewer.close()
            except Exception:  # noqa: BLE001
                pass
        self.preview_viewer = None
        self.preview_state = None
        self.preview_model = None
        self.template = None
        self.control = None
        self.goal_state = None
        self.state_0 = None
        self.state_1 = None
        self.solver = None
        self.model = None


@node(
    name="SO101ReachTask",
    component="runtime",
    category=_CATEGORY,
    description=(
        "Configure a simulation-only SO-ARM101 reaching curriculum for batched Newton/Warp "
        "reinforcement learning. This node never connects to or commands hardware."
    ),
    inputs={
        "trigger": AnyPort,
        "asset_path": Text(default=_ASSET_URI),
        "environment_count": Int(default=512),
        "episode_steps": Int(default=128),
        "simulation_hz": Int(default=120),
        "control_hz": Int(default=30),
        "action_scale_deg": Float(default=3.0),
        "success_tolerance_m": Float(default=0.025),
        "seed": Int(default=42),
    },
    outputs={
        "environment": Dict,
        "joint_names": List,
        "observation_dim": Int,
        "action_dim": Int,
        "report": Text,
    },
    primary_inputs=["trigger", "environment_count", "episode_steps"],
    primary_outputs=["environment", "report"],
)
def so101_reach_task(ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = environment_spec(
            asset_path=str(ctx.get("asset_path") or _ASSET_URI),
            environment_count=int(ctx.get("environment_count") or 512),
            episode_steps=int(ctx.get("episode_steps") or 128),
            simulation_hz=int(ctx.get("simulation_hz") or 120),
            control_hz=int(ctx.get("control_hz") or 30),
            action_scale_deg=float(ctx.get("action_scale_deg") or 3.0),
            success_tolerance_m=float(ctx.get("success_tolerance_m") or 0.025),
            seed=int(ctx.get("seed") or 42),
        )
        return {
            "environment": spec,
            "joint_names": list(SO101_JOINT_NAMES),
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "report": (
                f"SO-ARM101 reach task ready: {spec['environment_count']} simulated arms, "
                f"{spec['episode_steps']} control steps; physical motion remains disarmed"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "environment": {}, "joint_names": [], "observation_dim": 0, "action_dim": 0,
            "report": f"SO-ARM101 reach task FAILED: {type(exc).__name__}: {exc}",
        }
