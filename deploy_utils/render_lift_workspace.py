"""Render the Lift workspace rectangle from the configured top camera."""

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


def project_points(camera, points_world):
    k = camera.get_intrinsic_matrix()[0].detach().cpu().numpy()
    extrinsic = camera.get_extrinsic_matrix()[0].detach().cpu().numpy()
    points_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_cam = (extrinsic @ points_h.T).T[:, :3]
    uvw = (k @ points_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def main():
    parser = argparse.ArgumentParser(description="Render Lift workspace rectangle")
    parser.add_argument("--env-id", default="SO101LiftCube-v1")
    parser.add_argument("--robot-uids", default="xlerobot_right_head")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--output", default="/tmp/lift_workspace_range.png")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        num_envs=1,
        sim_backend="gpu",
        domain_randomization=False,
        domain_randomization_config={"initial_qpos_noise_scale": 0.0, "apply_overlay": True},
        sensor_configs={"width": args.width, "height": args.height},
    )
    try:
        obs, _ = env.reset(seed=0)
        uenv = env.unwrapped
        obs = uenv.get_obs()
        rgb = obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()

        # Lift samples around robot.pose.p + spawn_box_pos.
        robot_p = uenv.agent.robot.pose.p[0].detach().cpu().numpy()
        center = robot_p[:2] + np.asarray(uenv.spawn_box_pos, dtype=np.float32)
        half = np.asarray(uenv.spawn_box_half_size, dtype=np.float32)
        if half.size == 1:
            half = np.repeat(half, 2)

        xmin, xmax = center[0] - half[0], center[0] + half[0]
        ymin, ymax = center[1] - half[1], center[1] + half[1]
        rel_center = np.asarray(uenv.spawn_box_pos, dtype=np.float32)
        rel_xmin, rel_xmax = rel_center[0] - half[0], rel_center[0] + half[0]
        rel_ymin, rel_ymax = rel_center[1] - half[1], rel_center[1] + half[1]
        corners = np.array(
            [
                [xmin, ymin, 0.004],
                [xmax, ymin, 0.004],
                [xmax, ymax, 0.004],
                [xmin, ymax, 0.004],
            ],
            dtype=np.float32,
        )

        camera = uenv._sensors["base_camera"].camera
        pts = project_points(camera, corners).round().astype(np.int32)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.polylines(bgr, [pts], isClosed=True, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
        for idx, pt in enumerate(pts):
            cv2.circle(bgr, tuple(pt), 6, (0, 255, 255), -1)
            cv2.putText(
                bgr,
                str(idx + 1),
                tuple(pt + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        label = f"Lift spawn range rel x={rel_xmin:.2f}..{rel_xmax:.2f}m, y={rel_ymin:.2f}..{rel_ymax:.2f}m"
        cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(args.output, bgr)
        print(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
