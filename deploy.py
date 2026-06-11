"""
Deploy a trained RL agent on a real SO101/SO100 robot.

This script handles:
- Connecting to the real robot with either OpenCV (wrist) or RealSense camera
- Loading a trained checkpoint from local path or wandb
- Running evaluation episodes with keyboard controls

Usage:
    python deploy.py --checkpoint path/to/checkpoint.pt --env_id SO101ReachCube-v1
    python deploy.py --checkpoint wandb --env_id SO101ReachCube-v1  # Load from wandb
    python deploy.py --env_id SO101ReachCube-v1  # Random agent (no checkpoint)

Keyboard Controls:
    's' - Skip current episode
    'q' - Quit evaluation
"""

from dataclasses import dataclass
import random
from typing import Optional
from pathlib import Path
import sys
import signal
import select
import termios
import tty
import atexit
import time
import threading
import queue
import csv

import gymnasium as gym
import numpy as np
import torch
import tyro
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.visualization import tile_images
from transforms3d.euler import euler2quat


from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot

from train_squint import DeployAgent

# ============================================================
# ARGUMENTS
# ============================================================

@dataclass
class Args:
    checkpoint: Optional[str] = None
    """Path to checkpoint file, or 'wandb' to load from wandb. If None, uses random agent."""
    env_id: str = "SO101ReachCube-v1"
    """Environment ID (must match training environment)."""
    robot_uids: str = "so101"
    """Robot uid used by the simulation environment. Use 'xlerobot_right_head' for the URDF in ./urdf."""
    no_privileged_state: bool = False
    """For place tasks, hide object/bin privileged state so deployment matches B1 visual+qpos checkpoints."""
    obs_mode: str = "rgb+segmentation"
    """Observation mode for the environment."""
    control_mode: str = "pd_joint_target_delta_pos"
    """Control mode for the robot."""
    max_episode_steps: int = 200
    """Maximum steps per episode."""
    continuous_eval: bool = True
    """If True, runs without pausing. If False, waits for Enter at each step."""
    control_freq: Optional[int] = 30
    """Control frequency in Hz. Recommended: 15 or lower for safety."""
    action_scale: float = 0.15
    """Action scaling factor. Values < 1.0 make movements smaller/slower."""
    gripper_open_steps: int = 30
    """Force gripper open for the first N deployment steps to avoid premature closing."""
    gripper_open_action: float = 0.15
    """Scaled action command used while forcing the gripper open."""
    gripper_close_action_limit: float = 0.0
    """If > 0, caps negative gripper close commands to this magnitude, e.g. 0.04 for slow closing."""
    auto_stall_assist: bool = False
    """Trigger post-grasp assist when the policy stalls after the gripper has closed."""
    auto_stall_assist_once: bool = True
    """If enabled, auto stall assist triggers at most once per episode."""
    stall_assist_gripper_qpos: float = 0.45
    """Gripper qpos threshold used by auto stall assist. Lower means more closed."""
    stall_assist_action_norm: float = 0.025
    """Scaled arm-action norm threshold below which the policy is considered stalled."""
    stall_assist_min_step: int = 50
    """Earliest step where auto stall assist may trigger."""
    stall_assist_duration: int = 100
    """Number of deployment steps to keep auto stall assist active after triggering."""
    route_a_state_override: bool = False
    """Override shadow-sim item/bin poses for real deployment privileged-state validation."""
    route_a_item_xy: tuple[float, float] = (0.325, 0.0)
    """Fixed real item XY in the sim/world frame used for route A state override."""
    route_a_override_item_before_grasp: bool = False
    """If enabled, also override the shadow item XY before grasp. Leave disabled for free cube placement."""
    route_a_bin_xy: tuple[float, float] = (0.225, 0.0)
    """Fixed real bin XY in the sim/world frame used for route A state override."""
    route_a_follow_tcp_after_grasp: bool = True
    """After the gripper is closed, set the shadow item pose near TCP so item-to-bin state follows the grasp."""
    route_a_grasp_qpos: float = 0.45
    """Gripper qpos threshold below which route A treats the item as grasped/held."""
    route_a_tcp_item_z_offset: float = -0.015
    """Z offset from TCP used for the shadow item pose after grasp."""
    post_grasp_assist_start: int = 75
    """Start step for optional post-grasp return/lift assist. Set <0 to disable."""
    post_grasp_assist_end: int = -1
    """End step for optional post-grasp assist. Set <0 to keep assisting after start."""
    post_grasp_lift_action: float = 0.06
    """Minimum scaled shoulder lift action after post_grasp_assist_start."""
    post_grasp_wrist_action: float = 0.06
    """Minimum scaled wrist pitch action after post_grasp_assist_start."""
    post_grasp_close_action: float = -0.08
    """Maximum scaled gripper action after post_grasp_assist_start to keep the grasp closed."""
    route_a_place_assist: bool = False
    """Enable scripted route-A place validation after grasp/lift."""
    route_a_place_assist_carry_start: int = 150
    """Step where scripted carry-to-bin assist starts."""
    route_a_place_assist_release_start: int = 240
    """Step where scripted release starts."""
    route_a_place_assist_end: int = 285
    """Step where scripted place assist ends."""
    route_a_place_assist_base_action: float = 0.08
    """Scaled joint-0 action during carry. Flip sign if the robot moves away from the bin."""
    route_a_place_assist_elbow_action: float = -0.04
    """Scaled joint-2 action during carry."""
    route_a_place_assist_lift_action: float = 0.04
    """Scaled joint-1 action during carry to keep the cube lifted."""
    route_a_place_assist_wrist_action: float = 0.03
    """Scaled joint-3 action during carry to keep wrist from dropping."""
    route_a_place_assist_close_action: float = -0.06
    """Scaled gripper action during carry to keep holding the cube."""
    route_a_place_assist_release_action: float = 0.18
    """Scaled gripper action during release."""
    record_dir: Optional[str] = None
    """Directory to save episode recordings. If None, no recording."""
    record_resolution: int = 256
    """Resolution (HxW) for each frame in the recorded video. Use 128, 256, or 480."""
    policy_input_record_dir: Optional[str] = None
    """Directory to save the exact low-res RGB frames fed to the policy, upscaled for inspection."""
    policy_input_record_resolution: int = 256
    """Resolution for upscaled policy-input diagnostic videos."""
    real_crop_shift_x: int = 0
    """Pixel shift applied to the real camera square crop before resize. Positive crops farther right."""
    real_crop_shift_y: int = 0
    """Pixel shift applied to the real camera square crop before resize. Positive crops lower."""
    log_dir: Optional[str] = None
    """Directory to save per-step action/qpos logs. If None, no CSV logs."""
    debug: bool = False
    """If True, shows sim/real overlay visualization."""
    seed: int = 1
    """Random seed for reproducibility."""
    image_size: int = 128
    """HxW of input image to agent"""

    # Wandb checkpoint download settings (only used when checkpoint='wandb')
    wandb_entity: Optional[str] = None  # CHANGE THIS: your wandb username/entity
    wandb_project: str = "maniskill-so101"  # Your wandb project name
    wandb_agent_name: str = "squint"  # Agent name used in wandb
    wandb_version: str = "latest"  # Version of the checkpoint to download
    wandb_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)  # Seeds to check for best checkpoint


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def create_wrist_camera_preprocessor(sim_env, raw_frame_store=None, crop_shift_x=0, crop_shift_y=0):
    """Create a preprocessing function for the wrist camera images.

    Handles:
    - Cropping to square aspect ratio
    - Resizing to match simulation camera resolution

    Args:
        sim_env: The base simulation environment (unwrapped)

    Returns:
        Preprocessing function for sensor data
    """

    def preprocess(sensor_data, sensor_names=None):
        """Preprocess sensor data to match simulation format."""
        if sensor_names is None:
            sensor_names = list(sensor_data.keys())

        for sensor_name in sensor_names:
            sim_sensor_cfg = sim_env._sensor_configs[sensor_name]
            assert isinstance(sim_sensor_cfg, CameraConfig)

            target_h, target_w = sim_sensor_cfg.height, sim_sensor_cfg.width
            real_data = sensor_data[sensor_name]

            if "rgb" not in real_data:
                continue

            img = real_data["rgb"][0].numpy()

            # Crop to square aspect ratio
            h, w = img.shape[:2]
            crop_size = min(h, w)

            if h > w:
                max_offset = h - crop_size
                offset = int(np.clip((h - crop_size) // 2 + crop_shift_y, 0, max_offset))
                img = img[offset:offset + crop_size, :, :]
            elif w > h:
                max_offset = w - crop_size
                offset = int(np.clip((w - crop_size) // 2 + crop_shift_x, 0, max_offset))
                img = img[:, offset:offset + crop_size, :]

            if crop_shift_y != 0:
                transform = np.float32([[1, 0, 0], [0, 1, crop_shift_y]])
                img = cv2.warpAffine(
                    img,
                    transform,
                    (img.shape[1], img.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

            if raw_frame_store is not None:
                raw_frame_store["frame"] = img.copy()

            img = cv2.resize(img, (target_w, target_h))
            if raw_frame_store is not None:
                raw_frame_store["policy_frame"] = img.copy()
            real_data["rgb"] = common.to_tensor(img).unsqueeze(0)
            sensor_data[sensor_name] = real_data

        return sensor_data

    return preprocess


def setup_safe_exit(sim_env, real_env, real_agent, recorder=None, extra_recorders=None):
    """Register handlers for graceful shutdown on Ctrl+C or script exit."""
    def cleanup():
        print("\nCleaning up...")
        try:
            if recorder is not None:
                recorder.close()
        except Exception:
            pass
        if extra_recorders is not None:
            for extra_recorder in extra_recorders:
                try:
                    if extra_recorder is not None:
                        extra_recorder.close()
                except Exception:
                    pass
        try:
            if real_agent is not None:
                real_agent.reset(sim_env.unwrapped.agent.keyframes["rest"].qpos)
        except Exception:
            pass
        for env in [real_env, sim_env]:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    def signal_handler(sig=None, frame=None):
        print("\nCtrl+C detected. Exiting gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)


def overlay_envs(sim_env, real_env):
    """Overlay sim and real observations for visual debugging."""
    real_obs = real_env.unwrapped.get_obs()["sensor_data"]
    sim_obs = sim_env.unwrapped.get_obs()["sensor_data"]

    assert sorted(real_obs.keys()) == sorted(sim_obs.keys()), \
        f"Camera mismatch: real={real_obs.keys()}, sim={sim_obs.keys()}"

    overlaid_imgs = []
    for name in sim_obs:
        real_img = real_obs[name]["rgb"][0] / 255
        sim_img = sim_obs[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_img + 0.5 * sim_img)

    return tile_images(overlaid_imgs), real_img, sim_img


def print_timing_stats(timing_stats: dict, episode_num: int, target_freq: int):
    """Print timing statistics for an episode."""
    if not timing_stats["total"]:
        return

    print(f"\n--- Timing Stats (Episode {episode_num}) ---")
    print(f"Inference: {np.mean(timing_stats['inference'])*1000:.1f}ms avg, "
          f"{np.max(timing_stats['inference'])*1000:.1f}ms max")
    print(f"Step:      {np.mean(timing_stats['step'])*1000:.1f}ms avg, "
          f"{np.max(timing_stats['step'])*1000:.1f}ms max")
    print(f"Total:     {np.mean(timing_stats['total'])*1000:.1f}ms avg, "
          f"{np.max(timing_stats['total'])*1000:.1f}ms max")
    print(f"Achieved freq: {1/np.mean(timing_stats['total']):.1f} Hz (target: {target_freq} Hz)")


def silent_reset(env, seed=None, options=None):
    """Reset function without the 'Press enter' prompt."""
    env.sim_env.reset(seed=seed, options=options)
    env.agent.reset(qpos=env.base_sim_env.agent.robot.qpos.cpu().flatten())


def apply_route_a_state_override(real_env: Sim2RealEnv, real_agent: LeRobotRealAgent, args: Args):
    """Align privileged shadow-sim item/bin state with the real setup for route A validation."""
    if not args.route_a_state_override:
        return

    base_env = real_env.base_sim_env
    if not hasattr(base_env, "item") or not hasattr(base_env, "bin"):
        return

    device = base_env.device
    item_z = float(base_env.item_half_sizes.reshape(-1)[0].detach().cpu().item())
    bin_z = float(base_env.bin_thickness) / 2

    item_xy = torch.tensor(args.route_a_item_xy, dtype=torch.float32, device=device).reshape(1, 2)
    bin_xy = torch.tensor(args.route_a_bin_xy, dtype=torch.float32, device=device).reshape(1, 2)

    item_xyz = base_env.item.pose.p.clone()
    if args.route_a_override_item_before_grasp:
        item_xyz = torch.zeros((1, 3), dtype=torch.float32, device=device)
        item_xyz[:, :2] = item_xy
        item_xyz[:, 2] = item_z

    if args.route_a_follow_tcp_after_grasp:
        qpos_np = real_agent.qpos.detach().cpu().numpy().reshape(-1)
        if float(qpos_np[5]) <= args.route_a_grasp_qpos:
            tcp_pos = base_env.agent.tcp_pose.p.clone()
            item_xyz = tcp_pos
            item_xyz[:, 2] = torch.clamp(item_xyz[:, 2] + args.route_a_tcp_item_z_offset, min=item_z)

    bin_xyz = torch.zeros((1, 3), dtype=torch.float32, device=device)
    bin_xyz[:, :2] = bin_xy
    bin_xyz[:, 2] = bin_z

    quat = torch.tensor(euler2quat(0, 0, 0), dtype=torch.float32, device=device).reshape(1, 4)
    base_env.item.set_pose(Pose.create_from_pq(item_xyz, quat))
    base_env.bin.set_pose(Pose.create_from_pq(bin_xyz, quat))

    if hasattr(base_env, "goal_site"):
        goal_xyz = bin_xyz.clone()
        goal_xyz[:, 2] = float(base_env.bin_thickness) + item_z
        base_env.goal_site.set_pose(Pose.create_from_pq(goal_xyz, quat))


def refresh_route_a_obs(real_env: Sim2RealEnv, real_agent: LeRobotRealAgent, args: Args):
    """Apply route A shadow-state override and return wrapper-formatted observations."""
    apply_route_a_state_override(real_env, real_agent, args)
    if not args.route_a_state_override:
        return None

    if real_env._handle_wrappers:
        orig_env = real_env._last_wrapper.env
        real_env._last_wrapper.env = real_env._env_with_real_step_reset
        obs, _ = real_env._first_wrapper.reset()
        real_env._last_wrapper.env = orig_env
        return obs
    return real_env.get_obs()


def select_best_wandb_seed(entity: str, project: str, agent_name: str, env_id: str, seeds: list[int], version: str = "latest") -> Optional[int]:
    """Check wandb for checkpoints across seeds and return the one with highest eval/success_at_end."""
    import wandb
    api = wandb.Api()

    metrics_keys = ["eval/return", "eval/reward", "eval/success_at_end", "eval/success_once"]
    results = []

    for seed in seeds:
        artifact_name = f"{entity}/{project}/model_{agent_name}_{env_id}_{seed}:{version}"
        try:
            artifact = api.artifact(artifact_name)
            run = artifact.logged_by()
            metrics = {k: run.summary.get(k) for k in metrics_keys}
            results.append({"seed": seed, **metrics})
        except wandb.errors.CommError:
            print(f"  Seed {seed}: checkpoint not found ({artifact_name})")

    if not results:
        print("No checkpoints found for any seed!")
        return None

    print(f"\n{'Seed':<6} {'return':<12} {'reward':<12} {'success_end':<12} {'success_once':<12}")
    print("-" * 54)
    for r in results:
        print(f"{r['seed']:<6} {r.get('eval/return', 'N/A'):<12.4f} {r.get('eval/reward', 'N/A'):<12.4f} "
              f"{r.get('eval/success_at_end', 'N/A'):<12.4f} {r.get('eval/success_once', 'N/A'):<12.4f}")

    best = max(results, key=lambda x: x.get("eval/return", -float("inf")))
    print(f"\nSelected seed {best['seed']} (return: {best.get('eval/return', 'N/A'):.4f})")
    return best["seed"]


def extract_recording_frame(real_obs: dict, raw_frame_store: Optional[dict] = None) -> Optional[np.ndarray]:
    """Extract a BGR frame from the real observation for recording.

    Prefer the raw pre-resize camera frame when available, then fall back to the
    first RGB image in the observation dict.
    Returns a uint8 BGR numpy array, or None if no image found.
    """
    if raw_frame_store is not None and raw_frame_store.get("frame") is not None:
        img = raw_frame_store["frame"]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    for key in real_obs:
        if "rgb" in key or "image" in key:
            img = real_obs[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # Handle batched observations (B, H, W, C)
            if img.ndim == 4:
                img = img[0]
            # Convert to uint8 if normalized
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            # RGB -> BGR for cv2
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
    return None


def extract_policy_input_frame(raw_frame_store: Optional[dict] = None) -> Optional[np.ndarray]:
    """Return the exact low-res RGB policy input as a BGR frame for video writing."""
    if raw_frame_store is None or raw_frame_store.get("policy_frame") is None:
        return None
    img = raw_frame_store["policy_frame"]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ============================================================
# HELPER CLASSES
# ============================================================

class KeyboardController:
    """Non-blocking keyboard input handler for episode control."""

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *args):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_key(self) -> Optional[str]:
        """Check for keyboard input without blocking. Returns key or None."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class AsyncRecorder:
    """Non-blocking video recorder that writes frames in a background thread."""

    def __init__(self, output_dir: str, fps: int = 30, resolution: int = 256):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._write_loop, daemon=True)
        self._worker.start()
        self._episode_count = 0
        self._writer = None

    def push(self, frame: np.ndarray):
        """Push a BGR frame to be written. Non-blocking."""
        self._queue.put(("frame", frame))

    def end_episode(self):
        """Signal end of current episode. Starts a new video file on next push."""
        self._queue.put(("end_episode", None))

    def _write_loop(self):
        while True:
            msg_type, data = self._queue.get()

            if msg_type == "close":
                if self._writer is not None:
                    self._writer.release()
                break

            elif msg_type == "end_episode":
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                self._episode_count += 1

            elif msg_type == "frame":
                # Crop to square and resize to target resolution
                h, w = data.shape[:2]
                crop = min(h, w)
                y0 = (h - crop) // 2
                x0 = (w - crop) // 2
                data = data[y0:y0 + crop, x0:x0 + crop]
                if data.shape[0] != self.resolution or data.shape[1] != self.resolution:
                    data = cv2.resize(data, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

                if self._writer is None:
                    path = str(self.output_dir / f"episode_{self._episode_count}.mp4")
                    self._writer = cv2.VideoWriter(
                        path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (self.resolution, self.resolution),
                    )
                self._writer.write(data)

    @property
    def queue_size(self):
        return self._queue.qsize()

    def close(self):
        """Flush remaining frames and shut down the writer thread."""
        self._queue.put(("close", None))
        self._worker.join()


class EpisodeCSVLogger:
    """Simple per-episode CSV logger for real deployment diagnostics."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_count = 0
        self._file = None
        self._writer = None

    def start_episode(self):
        self.close_episode()
        path = self.output_dir / f"episode_{self._episode_count}.csv"
        self._file = path.open("w", newline="")
        fieldnames = (
            ["step"]
            + [f"action_{i}" for i in range(6)]
            + [f"scaled_action_{i}" for i in range(6)]
            + ["assist_active"]
            + [f"target_qpos_{i}" for i in range(6)]
            + [f"qpos_{i}" for i in range(6)]
        )
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log_step(
        self,
        step: int,
        action: np.ndarray,
        scaled_action: np.ndarray,
        assist_active: bool,
        target_qpos: Optional[torch.Tensor],
        qpos: torch.Tensor,
    ):
        if self._writer is None:
            return
        row = {"step": step}
        action = np.asarray(action).reshape(-1)
        scaled_action = np.asarray(scaled_action).reshape(-1)
        if target_qpos is None:
            target_qpos_np = np.full(6, np.nan)
        else:
            target_qpos_np = target_qpos.detach().cpu().numpy().reshape(-1)
        qpos_np = qpos.detach().cpu().numpy().reshape(-1)
        for i in range(6):
            row[f"action_{i}"] = float(action[i])
            row[f"scaled_action_{i}"] = float(scaled_action[i])
            row[f"target_qpos_{i}"] = float(target_qpos_np[i])
            row[f"qpos_{i}"] = float(qpos_np[i])
        row["assist_active"] = int(assist_active)
        self._writer.writerow(row)

    def close_episode(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def end_episode(self):
        self.close_episode()
        self._episode_count += 1


# ============================================================
# MAIN
# ============================================================

def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------------------------------
    # Phase 1: Robot & Environment Setup
    # --------------------------------------------------
    print("Setting up robot and environment...")

    real_robot = create_real_robot()
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode="sensors",
        max_episode_steps=args.max_episode_steps,
        robot_uids=args.robot_uids,
        domain_randomization=False,
        reward_mode="none",
        control_mode=args.control_mode,
        sensor_configs=dict(width=args.image_size, height=args.image_size)
    )
    if args.no_privileged_state:
        env_kwargs["privileged_state"] = False

    sim_env = gym.make(args.env_id, **env_kwargs)
    sim_env = FlattenRGBDObservationWrapper(sim_env, rgb=True, depth=False, state=True)

    # Async recorder for recording videos
    recorder = None
    if args.record_dir:
        recorder = AsyncRecorder(
            output_dir=args.record_dir,
            fps=args.control_freq or 30,
            resolution=args.record_resolution,
        )
    policy_input_recorder = None
    if args.policy_input_record_dir:
        policy_input_recorder = AsyncRecorder(
            output_dir=args.policy_input_record_dir,
            fps=args.control_freq or 30,
            resolution=args.policy_input_record_resolution,
        )
    csv_logger = EpisodeCSVLogger(args.log_dir) if args.log_dir else None

    raw_frame_store = {}
    preprocessor = create_wrist_camera_preprocessor(
        sim_env.unwrapped,
        raw_frame_store=raw_frame_store,
        crop_shift_x=args.real_crop_shift_x,
        crop_shift_y=args.real_crop_shift_y,
    )
    real_env = Sim2RealEnv(
        sim_env=sim_env,
        agent=real_agent,
        control_freq=args.control_freq,
        sensor_data_preprocessing_function=preprocessor,
        real_reset_function=silent_reset
    )

    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()
    route_a_obs = refresh_route_a_obs(real_env, real_agent, args)
    if route_a_obs is not None:
        real_obs = route_a_obs

    print("\nObservation shapes:")
    for k in sim_obs.keys():
        print(f"  {k}: sim={sim_obs[k].shape}, real={real_obs[k].shape}")

    setup_safe_exit(
        sim_env,
        real_env,
        real_agent,
        recorder=recorder,
        extra_recorders=[policy_input_recorder],
    )

    # --------------------------------------------------
    # Phase 2: Agent Loading
    # --------------------------------------------------
    print("\nLoading agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DeployAgent(sim_env, sample_obs=real_obs, target_image_size=args.image_size, device=device)

    if args.checkpoint:
        seed_to_use = args.seed
        if args.checkpoint == "wandb" and args.wandb_seeds:
            seed_to_use = select_best_wandb_seed(
                args.wandb_entity, args.wandb_project, args.wandb_agent_name,
                args.env_id, list(args.wandb_seeds), args.wandb_version
            )
            if seed_to_use is None:
                print("No valid checkpoints found. Using random agent.")
                args.checkpoint = None

        if args.checkpoint:
            checkpoint_config = {
                "wandb_entity": args.wandb_entity,
                "wandb_project_name": args.wandb_project,
                "agent_name": args.wandb_agent_name,
                "env_id": args.env_id,
                "seed": seed_to_use,
                "version": args.wandb_version
            }
            agent.load_checkpoint(args.checkpoint, checkpoint_config)
            print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided - using random agent")

    agent.to(device)

    # --------------------------------------------------
    # Phase 3: Debug Visualization Setup 
    # --------------------------------------------------
    if args.debug:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

        overlaid, real_img, sim_img = overlay_envs(sim_env, real_env)
        im1 = ax1.imshow(overlaid)
        ax1.set_title('Overlaid (Sim + Real)')
        im2 = ax2.imshow(sim_img)
        ax2.set_title('Simulation')
        im3 = ax3.imshow(real_img)
        ax3.set_title('Real')
        plt.tight_layout()

    # --------------------------------------------------
    # Phase 4: Evaluation Loop
    # --------------------------------------------------
    print("\n" + "=" * 40)
    print("KEYBOARD CONTROLS")
    print("  's' - Skip to next episode")
    print("  'q' - Quit evaluation")
    print("\nDeployment safety settings")
    print(f"  action_scale: {args.action_scale}")
    print(f"  gripper_open_steps: {args.gripper_open_steps}")
    print(f"  gripper_open_action: {args.gripper_open_action}")
    print(f"  gripper_close_action_limit: {args.gripper_close_action_limit}")
    print(f"  policy_input_record_dir: {args.policy_input_record_dir}")
    print(f"  policy_input_record_resolution: {args.policy_input_record_resolution}")
    print(f"  auto_stall_assist: {args.auto_stall_assist}")
    print(f"  auto_stall_assist_once: {args.auto_stall_assist_once}")
    print(f"  stall_assist_gripper_qpos: {args.stall_assist_gripper_qpos}")
    print(f"  stall_assist_action_norm: {args.stall_assist_action_norm}")
    print(f"  stall_assist_min_step: {args.stall_assist_min_step}")
    print(f"  stall_assist_duration: {args.stall_assist_duration}")
    print(f"  route_a_state_override: {args.route_a_state_override}")
    print(f"  route_a_item_xy: {args.route_a_item_xy}")
    print(f"  route_a_override_item_before_grasp: {args.route_a_override_item_before_grasp}")
    print(f"  route_a_bin_xy: {args.route_a_bin_xy}")
    print(f"  route_a_follow_tcp_after_grasp: {args.route_a_follow_tcp_after_grasp}")
    print(f"  real_crop_shift_x: {args.real_crop_shift_x}")
    print(f"  real_crop_shift_y: {args.real_crop_shift_y}")
    print(f"  post_grasp_assist_start: {args.post_grasp_assist_start}")
    print(f"  post_grasp_assist_end: {args.post_grasp_assist_end}")
    print(f"  route_a_place_assist: {args.route_a_place_assist}")
    print(f"  route_a_place_assist_carry_start: {args.route_a_place_assist_carry_start}")
    print(f"  route_a_place_assist_release_start: {args.route_a_place_assist_release_start}")
    print(f"  route_a_place_assist_end: {args.route_a_place_assist_end}")
    print("=" * 40 + "\n")

    episode_count = 0
    timing_stats = {"inference": [], "step": [], "total": []}

    with KeyboardController() as kb:
        while True:
            print("=============================")
            print(f"Episode {episode_count} - Press Enter to start, 'q' to quit")
            while (key := kb.check_key()) not in ('\n', '\r', 'q'):
                time.sleep(0.01)
            if key == 'q':
                print("[Quitting...]")
                break
            skip_episode = False
            if csv_logger:
                csv_logger.start_episode()
            auto_assist_until = -1
            auto_assist_triggered = False

            for step_idx in tqdm(range(args.max_episode_steps), desc="Steps"):
                key = kb.check_key()
                if key == 's':
                    print("\n[Skipping episode...]")
                    real_obs, _ = real_env.reset()
                    route_a_obs = refresh_route_a_obs(real_env, real_agent, args)
                    if route_a_obs is not None:
                        real_obs = route_a_obs
                    skip_episode = True
                    break
                elif key == 'q':
                    print("\n[Quitting...]")
                    skip_episode = True
                    break

                loop_start = time.perf_counter()

                route_a_obs = refresh_route_a_obs(real_env, real_agent, args)
                if route_a_obs is not None:
                    real_obs = route_a_obs
                obs = {k: v.to(device) for k, v in real_obs.items()}

                t0 = time.perf_counter()
                action = agent.get_action(obs)
                timing_stats["inference"].append(time.perf_counter() - t0)

                if not args.continuous_eval:
                    input("Press Enter to continue...")

                action = action.cpu().numpy()
                scaled_action = np.clip(action * args.action_scale, -1, 1)
                if step_idx < args.gripper_open_steps:
                    scaled_action[..., 5] = args.gripper_open_action
                elif args.gripper_close_action_limit > 0:
                    scaled_action[..., 5] = np.maximum(
                        scaled_action[..., 5], -args.gripper_close_action_limit
                    )
                if args.auto_stall_assist and step_idx >= args.stall_assist_min_step:
                    qpos_np = real_agent.qpos.detach().cpu().numpy().reshape(-1)
                    arm_action_norm = float(np.linalg.norm(scaled_action[..., :4].reshape(-1)))
                    gripper_qpos = float(qpos_np[5])
                    if (
                        (not args.auto_stall_assist_once or not auto_assist_triggered)
                        and
                        gripper_qpos <= args.stall_assist_gripper_qpos
                        and arm_action_norm <= args.stall_assist_action_norm
                        and step_idx >= auto_assist_until
                    ):
                        auto_assist_until = step_idx + args.stall_assist_duration
                        auto_assist_triggered = True

                post_grasp_assist_active = (
                    (
                        args.post_grasp_assist_start >= 0
                        and step_idx >= args.post_grasp_assist_start
                        and (
                            args.post_grasp_assist_end < 0
                            or step_idx < args.post_grasp_assist_end
                        )
                    )
                    or step_idx < auto_assist_until
                )
                if post_grasp_assist_active:
                    scaled_action[..., 1] = np.maximum(scaled_action[..., 1], args.post_grasp_lift_action)
                    scaled_action[..., 3] = np.maximum(scaled_action[..., 3], args.post_grasp_wrist_action)
                    scaled_action[..., 5] = np.minimum(scaled_action[..., 5], args.post_grasp_close_action)

                route_a_place_assist_active = (
                    args.route_a_place_assist
                    and step_idx >= args.route_a_place_assist_carry_start
                    and step_idx < args.route_a_place_assist_end
                )
                if route_a_place_assist_active:
                    if step_idx < args.route_a_place_assist_release_start:
                        scaled_action[..., 0] = args.route_a_place_assist_base_action
                        scaled_action[..., 1] = np.maximum(
                            scaled_action[..., 1], args.route_a_place_assist_lift_action
                        )
                        scaled_action[..., 2] = args.route_a_place_assist_elbow_action
                        scaled_action[..., 3] = np.maximum(
                            scaled_action[..., 3], args.route_a_place_assist_wrist_action
                        )
                        scaled_action[..., 5] = np.minimum(
                            scaled_action[..., 5], args.route_a_place_assist_close_action
                        )
                    else:
                        scaled_action[..., :4] = 0.0
                        scaled_action[..., 5] = args.route_a_place_assist_release_action

                t0 = time.perf_counter()
                real_obs, _, terminated, truncated, info = real_env.step(scaled_action)
                timing_stats["step"].append(time.perf_counter() - t0)

                if csv_logger:
                    csv_logger.log_step(
                        step_idx,
                        action,
                        scaled_action,
                        post_grasp_assist_active or route_a_place_assist_active,
                        real_agent._last_target_qpos,
                        real_agent.qpos,
                    )

                timing_stats["total"].append(time.perf_counter() - loop_start)

                # Async recording: push frame without blocking
                if recorder:
                    frame = extract_recording_frame(real_obs, raw_frame_store=raw_frame_store)
                    if frame is not None:
                        recorder.push(frame)
                if policy_input_recorder:
                    frame = extract_policy_input_frame(raw_frame_store=raw_frame_store)
                    if frame is not None:
                        policy_input_recorder.push(frame)

                if args.debug:
                    # Step sim env with zero action to update wrist camera (qpos already synced)
                    sim_env.step(np.zeros_like(scaled_action))

                    overlaid, real_img, sim_img = overlay_envs(sim_env, real_env)
                    im1.set_data(overlaid)
                    im2.set_data(sim_img)
                    im3.set_data(real_img)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)

            # If 'q' was pressed mid-episode, break out of the while loop too
            if key == 'q':
                break

            print_timing_stats(timing_stats, episode_count, args.control_freq)
            if recorder and recorder.queue_size > 0:
                print(f"  Recorder queue: {recorder.queue_size} frames pending write")
            if policy_input_recorder and policy_input_recorder.queue_size > 0:
                print(f"  Policy input recorder queue: {policy_input_recorder.queue_size} frames pending write")
            timing_stats = {"inference": [], "step": [], "total": []}

            # Signal end of episode to recorder
            if recorder:
                recorder.end_episode()
            if policy_input_recorder:
                policy_input_recorder.end_episode()
            if csv_logger:
                csv_logger.end_episode()

            episode_count += 1

            if not skip_episode:
                real_obs, _ = real_env.reset()
                route_a_obs = refresh_route_a_obs(real_env, real_agent, args)
                if route_a_obs is not None:
                    real_obs = route_a_obs

    # Cleanup
    print("\nReturning robot to rest position...")
    try:
        real_agent.reset(sim_env.unwrapped.agent.keyframes["rest"].qpos)
    except Exception as e:
        print(f"Warning: failed to reset robot to rest: {e}")

    if recorder:
        recorder.close()
    if policy_input_recorder:
        policy_input_recorder.close()

    print("Evaluation complete.")
    for env in [sim_env, real_env]:
        try:
            env.close()
        except Exception as e:
            print(f"Warning: error closing environment: {e}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
