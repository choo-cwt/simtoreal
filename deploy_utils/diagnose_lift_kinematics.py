import argparse
import os
import sys

import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import envs  # noqa: F401


JOINT_LABELS = [
    "Rotation_R",
    "Pitch_R",
    "Elbow_R",
    "Wrist_Pitch_R",
    "Wrist_Roll_R",
    "Jaw_R",
]


def as_list(tensor):
    return [round(float(x), 5) for x in tensor.detach().cpu().flatten()]


def print_state(env, label):
    agent = env.unwrapped.agent
    item = env.unwrapped.item
    tcp = agent.tcp_pos[0]
    item_pos = item.pose.p[0]
    fixed = agent.finger1_link.pose.p[0]
    moving = agent.finger2_link.pose.p[0]
    dist = torch.linalg.norm(item_pos - tcp)
    print(f"\n[{label}]")
    print(f"qpos        {as_list(agent.robot.get_qpos()[0])}")
    print(f"item_pos    {as_list(item_pos)}")
    print(f"tcp_pos     {as_list(tcp)}  dist={float(dist):.5f}")
    print(f"fixed_jaw   {as_list(fixed)}")
    print(f"moving_jaw  {as_list(moving)}")
    print(f"item-tcp    {as_list(item_pos - tcp)}")


def current_reward(env):
    info = env.unwrapped.evaluate()
    reward = env.unwrapped.compute_dense_reward(None, None, info)
    return float(reward[0].detach().cpu())


def step_action(env, action, n_steps):
    action = torch.as_tensor(action, device=env.unwrapped.device, dtype=torch.float32).reshape(1, -1)
    for _ in range(n_steps):
        env.step(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SO101LiftCube-v1")
    parser.add_argument("--robot-uids", default="xlerobot_right_head")
    parser.add_argument("--control-mode", default="pd_joint_target_delta_pos")
    parser.add_argument("--sim-backend", default="gpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--delta", type=float, default=1.0)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        control_mode=args.control_mode,
        obs_mode="state",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        domain_randomization=False,
        domain_randomization_config={"initial_qpos_noise_scale": 0.0, "apply_overlay": False},
    )

    try:
        env.reset(seed=args.seed)
        print_state(env, "reset")
        print(f"reward      {current_reward(env):.5f}")

        for joint_idx, joint_name in enumerate(JOINT_LABELS[:5]):
            for sign in (1.0, -1.0):
                env.reset(seed=args.seed)
                before_tcp = env.unwrapped.agent.tcp_pos[0].clone()
                before_item = env.unwrapped.item.pose.p[0].clone()
                before_dist = torch.linalg.norm(before_item - before_tcp)
                before_reward = current_reward(env)

                action = [0.0] * 6
                action[joint_idx] = sign * args.delta
                step_action(env, action, args.steps)

                after_tcp = env.unwrapped.agent.tcp_pos[0].clone()
                after_item = env.unwrapped.item.pose.p[0].clone()
                after_dist = torch.linalg.norm(after_item - after_tcp)
                after_reward = current_reward(env)
                print(
                    f"{joint_name:14s} {sign:+.0f}: "
                    f"tcp_delta={as_list(after_tcp - before_tcp)} "
                    f"dist {float(before_dist):.5f}->{float(after_dist):.5f} "
                    f"reward {before_reward:.5f}->{after_reward:.5f} "
                    f"qpos={as_list(env.unwrapped.agent.robot.get_qpos()[0])}"
                )

        for sign in (1.0, -1.0):
            env.reset(seed=args.seed)
            action = [0.0] * 6
            action[5] = sign * args.delta
            step_action(env, action, args.steps)
            print_state(env, f"gripper {sign:+.0f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
