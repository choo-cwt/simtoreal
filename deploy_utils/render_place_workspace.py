"""Render and check the Place workspace rectangle from the configured top camera."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import cv2
import gymnasium as gym
import numpy as np

import envs


def project_points(camera, points_world):
    k = camera.get_intrinsic_matrix()[0].detach().cpu().numpy()
    extrinsic = camera.get_extrinsic_matrix()[0].detach().cpu().numpy()
    points_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_cam = (extrinsic @ points_h.T).T[:, :3]
    uvw = (k @ points_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3], points_cam[:, 2]


def box_corners(center_xy, half_size_xy, height):
    x, y = center_xy
    hx, hy = half_size_xy
    return np.asarray(
        [
            [x + dx, y + dy, z]
            for dx in (-hx, hx)
            for dy in (-hy, hy)
            for z in (0.0, height)
        ],
        dtype=np.float32,
    )


def points_visible(uv, depth, width, height, margin):
    return (
        np.all(depth > 0)
        and np.all(uv[:, 0] >= margin)
        and np.all(uv[:, 0] <= width - 1 - margin)
        and np.all(uv[:, 1] >= margin)
        and np.all(uv[:, 1] <= height - 1 - margin)
    )


def main():
    parser = argparse.ArgumentParser(description="Render/check Place workspace visibility")
    parser.add_argument("--env-id", default="SO101PlaceCube-v1")
    parser.add_argument("--robot-uids", default="xlerobot_right_head")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--checks", type=int, default=100)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--output", default="/tmp/place_workspace_range.png")
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
        failures = []
        first_rgb = None
        first_camera = None
        for seed in range(args.checks):
            env.reset(seed=seed)
            uenv = env.unwrapped
            obs = uenv.get_obs()
            rgb = obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()
            camera = uenv._sensors["base_camera"].camera
            if first_rgb is None:
                first_rgb = rgb
                first_camera = camera

            item_xy = uenv.item.pose.p[0, :2].detach().cpu().numpy()
            item_half = float(uenv.item_half_sizes[0].detach().cpu().numpy())
            item_pts = box_corners(item_xy, (item_half, item_half), item_half * 2.0)
            item_uv, item_depth = project_points(camera, item_pts)

            bin_xy = uenv.bin.pose.p[0, :2].detach().cpu().numpy()
            bin_hx = float(uenv.bin_half_sizes_x[0].detach().cpu().numpy())
            bin_hy = float(uenv.bin_half_sizes_y[0].detach().cpu().numpy())
            bin_hz = float(uenv.bin_half_sizes_z[0].detach().cpu().numpy())
            bin_pts = box_corners(bin_xy, (bin_hx, bin_hy), bin_hz * 2.0)
            bin_uv, bin_depth = project_points(camera, bin_pts)

            if not points_visible(item_uv, item_depth, args.width, args.height, args.margin):
                failures.append((seed, "item", item_xy.tolist()))
            if not points_visible(bin_uv, bin_depth, args.width, args.height, args.margin):
                failures.append((seed, "bin", bin_xy.tolist()))

        uenv = env.unwrapped
        robot_p = uenv.agent.robot.pose.p[0].detach().cpu().numpy()
        center = robot_p[:2] + np.asarray(uenv.spawn_box_pos, dtype=np.float32)
        half = np.asarray(uenv.spawn_box_half_size, dtype=np.float32)
        if half.size == 1:
            half = np.repeat(half, 2)
        xmin, xmax = center[0] - half[0], center[0] + half[0]
        ymin, ymax = center[1] - half[1], center[1] + half[1]
        rect = np.asarray(
            [[xmin, ymin, 0.004], [xmax, ymin, 0.004], [xmax, ymax, 0.004], [xmin, ymax, 0.004]],
            dtype=np.float32,
        )
        rect_uv, _ = project_points(first_camera, rect)
        bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
        cv2.polylines(bgr, [rect_uv.round().astype(np.int32)], True, (0, 0, 255), 4, cv2.LINE_AA)
        label = f"Place range x={xmin:.2f}..{xmax:.2f} y={ymin:.2f}..{ymax:.2f}, failures={len(failures)}"
        cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(args.output, bgr)

        print(args.output)
        print(label)
        if failures:
            print("visibility failures:")
            for failure in failures[:20]:
                print(failure)
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
