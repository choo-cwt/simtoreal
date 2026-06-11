"""Render XLeRobot start-pose wrist roll variants for quick gripper orientation checks."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import cv2
import gymnasium as gym
import numpy as np
import torch

import envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/wrist_roll_variants.png")
    args = parser.parse_args()

    env = gym.make(
        "SO101LiftCube-v1",
        robot_uids="xlerobot_right_head",
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="gpu",
        domain_randomization=False,
        domain_randomization_config={"initial_qpos_noise_scale": 0.0, "apply_overlay": True},
        sensor_configs={"width":256, "height":256},
        human_render_camera_configs={"width":256, "height":256},
    )
    try:
        env.reset(seed=0)
        uenv = env.unwrapped
        base_qpos = np.array([0, 0, 0, np.pi / 2, 0, 60 * np.pi / 180], dtype=np.float32)
        variants = [
            ("roll -180", -np.pi),
            ("roll -90", -np.pi / 2),
            ("roll 0", 0.0),
            ("roll +90", np.pi / 2),
            ("roll +180", np.pi),
        ]
        images = []
        for label, roll in variants:
            qpos = torch.tensor(base_qpos, device=uenv.device).unsqueeze(0)
            qpos[:, 4] = roll
            uenv.agent.robot.set_qpos(qpos)
            uenv.agent.robot.set_pose(uenv.agent.robot.pose)
            if uenv.gpu_sim_enabled:
                uenv.scene._gpu_apply_all()
                uenv.scene._gpu_fetch_all()
            frame = env.render()[0].detach().cpu().numpy()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            images.append(bgr)
        out = np.hstack(images)
        cv2.imwrite(args.output, out)
        print(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
