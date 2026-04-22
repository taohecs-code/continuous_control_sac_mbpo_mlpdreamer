import argparse
import random

import gymnasium as gym
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from sac.sac_agent import SACAgent, SACConfig


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = float(reward)
        self.next_state[self.ptr] = next_state
        self.done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
        )


def evaluate_policy(agent, env_name, eval_episodes, seed):
    env = gym.make(env_name)
    reward_sum = 0.0
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
        reward_sum += episode_reward
    env.close()
    return reward_sum / eval_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Clean SAC Runner")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Total env steps")
    parser.add_argument("--prefill_steps", type=int, default=5000, help="Random actions before training")
    parser.add_argument("--eval_freq", type=int, default=2000, help="Eval every N steps")
    parser.add_argument("--batch_size", type=int, default=256, help="SAC batch size")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Replay capacity")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="sac-student-version", help="WandB project")

    parser.add_argument("--lr", type=float, default=3e-4, help="Actor & Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount")
    parser.add_argument("--tau", type=float, default=0.005, help="Target Polyak coefficient")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy coefficient (if not auto)")
    parser.add_argument("--mlp_depth", type=int, default=2, help="MLP layer count for actor/critic")
    parser.add_argument("--auto_alpha", action="store_true", help="Learn entropy temperature")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=f"clean-sac-{args.env}-seed{args.seed}",
            config=vars(args),
        )

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    sac_cfg = SACConfig(
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        mlp_depth=args.mlp_depth,
        action_scale=action_scale,
        action_bias=action_bias,
    )

    agent = SACAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        cfg=sac_cfg,
    )

    replay_buffer = ReplayBuffer(
        state_dim=obs_dim,
        action_dim=action_dim,
        max_size=args.buffer_size,
        device=device,
    )

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    print(f"SAC on {args.env} | seed={args.seed} | device={device}")

    for step in range(1, args.max_steps + 1):
        if step <= args.prefill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
        episode_steps += 1

        done_for_learning = float(terminated or truncated)
        replay_buffer.add(obs, action, reward, next_obs, done_for_learning)

        obs = next_obs

        if terminated or truncated:
            episode_count += 1
            if args.wandb and wandb is not None:
                wandb.log(
                    {"train/episode_reward": episode_reward, "train/episode_steps": episode_steps},
                    step=step,
                )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        if step > args.prefill_steps and replay_buffer.size >= args.batch_size:
            stats = agent.train(replay_buffer, batch_size=args.batch_size)
            if args.wandb and wandb is not None and step % 100 == 0:
                wandb.log({f"sac/{k}": v for k, v in stats.items()}, step=step)

        if step % args.eval_freq == 0 and step > args.prefill_steps:
            eval_reward = evaluate_policy(agent, args.env, eval_episodes=5, seed=args.seed + 100)
            print(f"Step {step}/{args.max_steps} | Eval Reward: {eval_reward:.2f}")
            if args.wandb and wandb is not None:
                wandb.log({"eval/reward": eval_reward}, step=step)

    env.close()
    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
