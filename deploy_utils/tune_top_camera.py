"""Live tuner for the XLeRobot fixed top camera.

Side-by-side view: Real | Sim | Edge overlay.
Trackbars adjust the simulated top camera local orientation and FOV.
Keys: p=print params, r=rest pose, s=start pose, f=apply FOV, q=quit.
"""

import argparse
import atexit
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import cv2
import gymnasium as gym
import numpy as np
import torch
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qinverse, qmult

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot

import envs


class LiveTopCameraTuner:
    BASE_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    def __init__(
        self,
        env_id: str,
        sim_width: int = 480,
        sim_height: int = 480,
        sim_backend: str = "gpu",
        head_port: str | None = None,
        head_pan_id: int = 9,
        head_tilt_id: int = 10,
        use_head_servos: bool = False,
        head_pan_center: float = 2048.0,
        head_tilt_center: float = 2048.0,
        head_pan_sign: float = 1.0,
        head_tilt_sign: float = 1.0,
    ):
        self.env_id = env_id
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.sim_backend = sim_backend
        self.head_port = head_port
        self.head_pan_id = head_pan_id
        self.head_tilt_id = head_tilt_id
        self.use_head_servos = use_head_servos
        self.head_pan_center = head_pan_center
        self.head_tilt_center = head_tilt_center
        self.head_pan_sign = head_pan_sign
        self.head_tilt_sign = head_tilt_sign
        self.head_servo_raw = None

        self.yaw_deg = 0.0
        self.pitch_deg = 60.0
        self.roll_deg = 0.0
        self.fov = 60.0
        self._last_fov = self.fov
        self._fov_pending = False

        self.sim_env = None
        self.real_robot = None
        self.real_agent = None

        self._create_sim_env()
        self._setup_real_robot()
        self._extract_camera_params()
        self.refresh_head_servos(apply_offsets=self.use_head_servos)
        self._setup_exit()
        self._setup_ui()

    def _current_q(self):
        offset_q = euler2quat(
            np.deg2rad(self.roll_deg),
            np.deg2rad(self.pitch_deg),
            np.deg2rad(self.yaw_deg),
            axes="sxyz",
        )
        q = qmult(self.BASE_Q, offset_q)
        return [float(v) for v in q]

    def _create_sim_env(self, preserve_fov=False):
        desired_fov = self.fov if preserve_fov else None
        if self.sim_env is not None:
            self.sim_env.close()

        sensor_configs = {"width": self.sim_width, "height": self.sim_height}
        if preserve_fov and desired_fov is not None:
            sensor_configs["fov"] = np.deg2rad(desired_fov)

        self.sim_env = gym.make(
            self.env_id,
            robot_uids="xlerobot_right_head",
            obs_mode="rgb+segmentation",
            render_mode="sensors",
            num_envs=1,
            sim_backend=self.sim_backend,
            domain_randomization=False,
            domain_randomization_config={"initial_qpos_noise_scale": 0.0, "apply_overlay": True},
            sensor_configs=sensor_configs,
        )
        self.sim_env = FlattenRGBDObservationWrapper(self.sim_env, rgb=True, depth=False, state=True)
        self.sim_env.reset(seed=0)
        if preserve_fov and desired_fov is not None:
            self.fov = desired_fov
        self._last_fov = self.fov

    def _extract_camera_params(self):
        env = self.sim_env.unwrapped
        if hasattr(env, "TOP_CAMERA_FOV"):
            self.fov = float(np.rad2deg(env.TOP_CAMERA_FOV))
            self._last_fov = self.fov
        if hasattr(env, "TOP_CAMERA_LOCAL_Q"):
            offset_q = qmult(qinverse(self.BASE_Q), np.asarray(env.TOP_CAMERA_LOCAL_Q, dtype=np.float64))
            roll, pitch, yaw = quat2euler(offset_q, axes="sxyz")
            self.roll_deg = float(np.rad2deg(roll))
            self.pitch_deg = float(np.rad2deg(pitch))
            self.yaw_deg = float(np.rad2deg(yaw))

    def _setup_real_robot(self):
        self.real_robot = create_real_robot()
        self.real_robot.connect()
        self.real_agent = LeRobotRealAgent(self.real_robot)

    def _read_head_servos(self):
        if not self.head_port:
            return None

        bus = FeetechMotorsBus(
            port=self.head_port,
            motors={
                "head_pan": Motor(self.head_pan_id, "sts3215", MotorNormMode.RANGE_M100_100),
                "head_tilt": Motor(self.head_tilt_id, "sts3215", MotorNormMode.RANGE_M100_100),
            },
        )
        try:
            bus.connect()
            return bus.sync_read("Present_Position", normalize=False, num_retry=3)
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)

    @staticmethod
    def _raw_ticks_to_deg(raw, center):
        return (float(raw) - float(center)) * 360.0 / 4095.0

    def refresh_head_servos(self, apply_offsets=False):
        pos = self._read_head_servos()
        if pos is None:
            if apply_offsets:
                print("Head servo read skipped: no --head-port set")
            return

        self.head_servo_raw = pos
        pan_deg = self._raw_ticks_to_deg(pos["head_pan"], self.head_pan_center)
        tilt_deg = self._raw_ticks_to_deg(pos["head_tilt"], self.head_tilt_center)
        print(
            "Head servos:",
            f"pan raw={pos['head_pan']:.0f} approx={pan_deg:.2f}deg,",
            f"tilt raw={pos['head_tilt']:.0f} approx={tilt_deg:.2f}deg",
        )

        if apply_offsets:
            self.yaw_deg = self.head_pan_sign * pan_deg
            self.pitch_deg = 60.0 + self.head_tilt_sign * tilt_deg
            print(f"Applied head servo estimate: yaw={self.yaw_deg:.2f}, pitch={self.pitch_deg:.2f}")

    def _move_real_to_sim_pose(self):
        qpos = self.sim_env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.squeeze()
        env = self.sim_env.unwrapped
        env.agent.robot.set_qpos(qpos.unsqueeze(0))
        if env.gpu_sim_enabled:
            env.scene._gpu_apply_all()
        self.real_agent.reset(qpos)

    def _flip_sim_wrist_roll(self):
        qpos = self.sim_env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.squeeze()
        qpos[4] = torch.remainder(qpos[4] + np.pi + np.pi, 2 * np.pi) - np.pi
        env = self.sim_env.unwrapped
        env.agent.robot.set_qpos(qpos.unsqueeze(0))
        if env.gpu_sim_enabled:
            env.scene._gpu_apply_all()

    def _apply_camera_params(self):
        env = self.sim_env.unwrapped
        env.TOP_CAMERA_LOCAL_Q = self._current_q()
        env.TOP_CAMERA_FOV = np.deg2rad(self.fov)

    def _get_real_image(self):
        self.real_agent.capture_sensor_data()
        obs = self.real_agent.get_sensor_data()
        if "base_camera" not in obs or "rgb" not in obs["base_camera"]:
            return None
        rgb = obs["base_camera"]["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        h, w = rgb.shape[:2]
        if h != w:
            s = min(h, w)
            c = (max(h, w) - s) // 2
            rgb = rgb[c : c + s, :, :] if h > w else rgb[:, c : c + s, :]
        rgb = cv2.resize(rgb, (self.sim_width, self.sim_height))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _get_sim_image(self):
        self._apply_camera_params()
        obs = self.sim_env.unwrapped.get_obs()
        sensor = obs["sensor_data"]
        if "base_camera" not in sensor or "rgb" not in sensor["base_camera"]:
            return None
        rgb = sensor["base_camera"]["rgb"][0].cpu().numpy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _make_comparison(self, real, sim):
        if real is None or sim is None:
            return None
        h, w = real.shape[:2]
        sim_r = cv2.resize(sim, (w, h))
        blended = cv2.addWeighted(real, 0.5, sim_r, 0.5, 0)
        comp = np.hstack([real, sim_r, blended])

        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [("Real", 10), ("Sim", w + 10), ("Blend", 2 * w + 10)]
        for text, x in labels:
            cv2.putText(comp, text, (x, 50), font, 1.5, (0, 0, 0), 5)
            cv2.putText(comp, text, (x, 50), font, 1.5, (255, 255, 255), 3)

        params = (
            f"yaw={self.yaw_deg:.1f} pitch={self.pitch_deg:.1f} roll={self.roll_deg:.1f} "
            f"fov={self.fov:.0f}"
        )
        cv2.putText(comp, params, (10, comp.shape[0] - 15), font, 0.7, (0, 0, 0), 3)
        cv2.putText(comp, params, (10, comp.shape[0] - 15), font, 0.7, (255, 255, 255), 2)
        if self.head_servo_raw is not None:
            head_text = f"head raw pan={self.head_servo_raw['head_pan']:.0f} tilt={self.head_servo_raw['head_tilt']:.0f}"
            cv2.putText(comp, head_text, (10, comp.shape[0] - 45), font, 0.7, (0, 0, 0), 3)
            cv2.putText(comp, head_text, (10, comp.shape[0] - 45), font, 0.7, (255, 255, 255), 2)
        return comp

    def _setup_ui(self):
        self.win = "Top Camera Tuner | p:print r:rest s:start f:FOV q:quit"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Yaw offset", self.win, int(self.yaw_deg + 180), 360, lambda v: setattr(self, "yaw_deg", v - 180.0))
        cv2.createTrackbar("Pitch offset", self.win, int(self.pitch_deg + 90), 180, lambda v: setattr(self, "pitch_deg", v - 90.0))
        cv2.createTrackbar("Roll offset", self.win, int(self.roll_deg + 90), 180, lambda v: setattr(self, "roll_deg", v - 90.0))
        cv2.createTrackbar("FOV", self.win, int(self.fov), 120, self._on_fov)

    def _on_fov(self, val):
        new = max(10, val)
        if new != self.fov:
            self.fov = new
            self._fov_pending = True

    def _setup_exit(self):
        def cleanup(sig=None, frame=None):
            try:
                self.real_robot and self.real_robot.disconnect()
            except Exception:
                pass
            try:
                self.sim_env and self.sim_env.close()
            except Exception:
                pass
            if sig is not None:
                sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        atexit.register(cleanup)
        self._cleanup = cleanup

    def print_params(self):
        q = self._current_q()
        print(f"\n{'=' * 60}")
        print("Top camera params for TopCameraEnv (envs/base_random_env.py):")
        print(f"  TOP_CAMERA_FOV = np.deg2rad({self.fov:.1f})")
        if hasattr(self.sim_env.unwrapped, "TOP_CAMERA_LOCAL_P"):
            print("  TOP_CAMERA_LOCAL_P = " + str(self.sim_env.unwrapped.TOP_CAMERA_LOCAL_P))
        print("  TOP_CAMERA_LOCAL_Q = " + str([round(v, 6) for v in q]))
        print(f"{'=' * 60}\n")

    def print_camera_mount(self):
        env = self.sim_env.unwrapped
        link = env.agent.robot.links_map["head_camera_rgb_frame"]
        print("\nUsing sensor 'base_camera' mounted to URDF link 'head_camera_rgb_frame'.")
        print("head_camera_rgb_frame p =", link.pose.p[0].detach().cpu().numpy().round(4).tolist())
        print("TOP_CAMERA_LOCAL_Q =", [round(v, 6) for v in self._current_q()])

    def run(self):
        print("\nControls:")
        print("  p  - Print current top camera parameters")
        print("  b  - Print camera mount/link confirmation")
        print("  h  - Refresh head servo raw positions")
        print("  r  - Move sim+real to rest pose")
        print("  s  - Move sim+real to start pose")
        print("  x  - Flip sim wrist_roll 180 deg for diagnosis only")
        print("  f  - Apply pending FOV change")
        print("  q  - Quit")
        print("  Trackbars - Adjust Yaw/Pitch/Roll/FOV\n")

        while True:
            comp = self._make_comparison(self._get_real_image(), self._get_sim_image())
            if comp is not None:
                if self.fov != self._last_fov:
                    text = f"FOV: {self._last_fov:.0f}->{self.fov:.0f} (press 'f')"
                    cv2.putText(comp, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(comp, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow(self.win, comp)
            else:
                cv2.imshow(self.win, np.zeros((480, 1440, 3), dtype=np.uint8))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                self.print_params()
            elif key == ord("b"):
                self.print_camera_mount()
            elif key == ord("h"):
                try:
                    self.refresh_head_servos(apply_offsets=False)
                except Exception as e:
                    print(f"Head servo read error: {e}")
            elif key == ord("r"):
                try:
                    rest_qpos = self.sim_env.unwrapped.agent.keyframes["rest"].qpos
                    qpos = rest_qpos if isinstance(rest_qpos, torch.Tensor) else torch.tensor(rest_qpos, dtype=torch.float32)
                    if qpos.dim() == 1:
                        qpos = qpos.unsqueeze(0)
                    env = self.sim_env.unwrapped
                    env.agent.robot.set_qpos(qpos)
                    if env.gpu_sim_enabled:
                        env.scene._gpu_apply_all()
                    self.real_agent.reset(qpos.squeeze())
                    print("Moved sim+real to rest pose")
                except Exception as e:
                    print(f"Rest pose error: {e}")
            elif key == ord("s"):
                try:
                    self.sim_env.reset(seed=0)
                    self._move_real_to_sim_pose()
                    print("Moved sim+real to start pose")
                except Exception as e:
                    print(f"Start pose error: {e}")
            elif key == ord("x"):
                try:
                    self._flip_sim_wrist_roll()
                    print("Flipped sim wrist_roll by 180 deg (real robot unchanged)")
                except Exception as e:
                    print(f"Flip sim wrist_roll error: {e}")
            elif key == ord("f") and self._fov_pending:
                self._create_sim_env(preserve_fov=True)
                self._fov_pending = False

        cv2.destroyAllWindows()
        self._cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live XLeRobot fixed top camera tuning")
    parser.add_argument("--env-id", default="SO101LiftCube-v1")
    parser.add_argument("--sim-width", type=int, default=480)
    parser.add_argument("--sim-height", type=int, default=480)
    parser.add_argument("--sim-backend", default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--head-port", default=None, help="Optional Feetech bus port for head servos, e.g. /dev/ttyACM1")
    parser.add_argument("--head-pan-id", type=int, default=9)
    parser.add_argument("--head-tilt-id", type=int, default=10)
    parser.add_argument("--use-head-servos", action="store_true", help="Initialize yaw/pitch from current head servo raw ticks")
    parser.add_argument("--head-pan-center", type=float, default=2048.0)
    parser.add_argument("--head-tilt-center", type=float, default=2048.0)
    parser.add_argument("--head-pan-sign", type=float, default=1.0)
    parser.add_argument("--head-tilt-sign", type=float, default=1.0)
    args = parser.parse_args()
    LiveTopCameraTuner(
        args.env_id,
        args.sim_width,
        args.sim_height,
        args.sim_backend,
        head_port=args.head_port,
        head_pan_id=args.head_pan_id,
        head_tilt_id=args.head_tilt_id,
        use_head_servos=args.use_head_servos,
        head_pan_center=args.head_pan_center,
        head_tilt_center=args.head_tilt_center,
        head_pan_sign=args.head_pan_sign,
        head_tilt_sign=args.head_tilt_sign,
    ).run()
