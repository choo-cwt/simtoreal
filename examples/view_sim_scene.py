import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import gymnasium as gym
from mani_skill.utils import sapien_utils

import envs


def main():
    parser = argparse.ArgumentParser(description="Open the ManiSkill/SAPIEN viewer for a Squint task.")
    parser.add_argument("--env-id", default="SO101LiftCube-v1")
    parser.add_argument("--robot-uids", default="xlerobot_right_head")
    parser.add_argument("--sim-backend", default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eye", type=float, nargs=3, default=[1.1, -1.2, 0.8])
    parser.add_argument("--target", type=float, nargs=3, default=[0.05, 0.0, 0.1])
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend=args.sim_backend,
        domain_randomization=False,
        domain_randomization_config={"initial_qpos_noise_scale": 0.0, "apply_overlay": False},
        viewer_camera_configs={
            "shader_pack": "default",
            "pose": sapien_utils.look_at(args.eye, args.target),
        },
    )
    env.reset(seed=args.seed)
    viewer = env.unwrapped.render_human()

    print("SAPIEN viewer opened. Mouse: orbit/pan/zoom. Press q in the viewer to quit.")
    try:
        while True:
            env.unwrapped.render_human()
            if viewer.window.key_press("q"):
                break
            time.sleep(1 / 60)
    finally:
        env.close()


if __name__ == "__main__":
    main()
