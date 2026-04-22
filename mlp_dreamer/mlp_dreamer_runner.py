import argparse
import random
import numpy as np
import torch
import gymnasium as gym

try:
    import wandb
except ImportError:
    wandb = None

from mlp_dreamer.mlp_dreamer_agent import DreamerConfig, DreamerAgent


class ReplayBuffer:
    """
    Circular Buffer.
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device) 

        # pre-allocate memory, greatly improve speed
        # gymnasium environment output is numpy array or python native float
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)
        self.episode_end = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated, episode_end):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.terminated[self.ptr] = float(terminated)
        self.episode_end[self.ptr] = float(episode_end)
        
        # move pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_sequences(self, batch_size, seq_len):
        """
        Provide sequence of length seq_len from the buffer for dreamer.
        (burn in and real time interaction)

        If episodes are shorter than seq_len (e.g. InvertedDoublePendulum), the
        effective length is reduced to the longest window with no episode_end inside.
        """
        requested = seq_len
        end_signals_flat = self.episode_end[: self.size, 0].astype(np.int32)
        end_csum = np.concatenate([[0], np.cumsum(end_signals_flat, dtype=np.int64)])

        effective = seq_len
        valid_starts = np.empty(0, dtype=np.int64)

        # if buffer is too small, reduce sequence length until effective
        while effective >= 1:
            # valid data length in self.size
            if self.size < effective + 1:
                effective -= 1
                continue
            safe_start = self.size - effective
            safe_check_window = effective - 1
            if safe_check_window <= 0:
                valid_starts = np.arange(safe_start, dtype=np.int64)
            else:
                end_count = end_csum[safe_check_window:] - end_csum[:-safe_check_window]
                valid_starts = np.where(end_count[:safe_start] == 0)[0]
            if valid_starts.size > 0:
                break
            effective -= 1

        if valid_starts.size == 0:
            raise ValueError(
                "No valid starting points for any sequence length up to %d (buffer size %d)"
                % (requested, self.size)
            )

        seq_len = effective

        starts_idx = np.random.choice(valid_starts, size=batch_size, replace=valid_starts.size < batch_size)

        seq_idx = starts_idx[:, None] + np.arange(seq_len)[None, :]

        return (
            torch.FloatTensor(self.state[seq_idx]).to(self.device),
            torch.FloatTensor(self.action[seq_idx]).to(self.device),
            torch.FloatTensor(self.reward[seq_idx]).to(self.device),
            torch.FloatTensor(self.next_state[seq_idx]).to(self.device),
            torch.FloatTensor(self.terminated[seq_idx]).to(self.device),
            torch.FloatTensor(self.episode_end[seq_idx]).to(self.device),
        )

def evaluate_policy(agent, env, eval_episodes, seed):
    """
    Evaluate the policy on the environment for some evaluation steps.
    (Learning curves)
    """
    env = gym.make(env)

    reward_sum = 0.0

    for ep in range(eval_episodes):

        # fix the seed
        obs, _ = env.reset(seed=seed + ep)

        agent.reset_episode()
        done = False
        episode_reward = 0.0
        while not done:

            action = agent.select_action_real_env(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            
        reward_sum += episode_reward
        
    env.close()
    return reward_sum / eval_episodes

def parse_args():
    """
    CLI Parser.
    """
    parser = argparse.ArgumentParser(description="Clean Dreamer Runner")
    
    # --- Macro Experiment Controls ---
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Total environment steps")
    parser.add_argument("--prefill_steps", type=int, default=1000, help="Random exploration steps before training")
    parser.add_argument("--eval_freq", type=int, default=2000, help="Evaluation frequency (in steps)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for Dreamer training")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="dreamer-student-version", help="WandB project name")
    
    # --- Core Algorithm Hyperparameters (Overrides DreamerConfig defaults) ---
    # Note: default=None means "use the default value from DreamerConfig if not specified in CLI"
    parser.add_argument("--horizon", type=int, default=None, help="Imagination horizon")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--lambda_", type=float, default=None, help="TD(lambda) target decay")
    parser.add_argument("--kl_beta", type=float, default=None, help="KL divergence weight")
    parser.add_argument("--wm_lr", type=float, default=None, help="World Model learning rate")
    parser.add_argument("--actor_lr", type=float, default=None, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=None, help="Critic learning rate")
    
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
            name=f"clean-dreamer-{args.env}-seed{args.seed}",
            config=vars(args),
        )
    
    env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cfg = DreamerConfig()
    if args.horizon is not None: cfg.horizon = args.horizon
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.lambda_ is not None: cfg.lambda_ = args.lambda_
    if args.kl_beta is not None: cfg.kl_beta = args.kl_beta
    if args.wm_lr is not None: cfg.wm_lr = args.wm_lr
    if args.actor_lr is not None: cfg.actor_lr = args.actor_lr
    if args.critic_lr is not None: cfg.critic_lr = args.critic_lr

    agent = DreamerAgent(obs_dim, action_dim, cfg=cfg, device=device)
    buffer = ReplayBuffer(obs_dim, action_dim, max_size=100_000, device=device)

    obs, _ = env.reset(seed=args.seed)
    agent.reset_episode()
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    print(f"Dreamer on {args.env} with seed {args.seed}")

    # # start from 1 to avoid evaluation at step 0
    for step in range(1, args.max_steps + 1): 
        # random prefill
        if step <= args.prefill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action_real_env(obs, deterministic=False)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        episode_reward += float(reward)
        episode_steps += 1
        
        buffer.add(obs, action, reward, next_obs, terminated, terminated or truncated)
        obs = next_obs

        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} finished after {episode_steps} steps with reward {episode_reward:.2f}")
            if args.wandb and wandb is not None:
                wandb.log({"train/episode_reward": episode_reward, "train/episode_steps": episode_steps}, step=step)

            obs, _ = env.reset()
            agent.reset_episode()
            episode_reward = 0.0
            episode_steps = 0

        # train networks
        if step > args.prefill_steps and buffer.size >= args.seq_len + 1:
            # sample a batch of sequences
            b_obs, b_act, b_rew, _, b_term, b_end = buffer.sample_sequences(args.batch_size, args.seq_len)

            # train wm
            wm_metrics = agent.train_world_model_from_buffer(b_obs, b_act, b_rew, b_term, b_end)
            
            # train Actor-Critic
            ac_metrics = agent.train_actor_critic_in_dreams(b_obs, b_act)

            if args.wandb and wandb is not None and step % 100 == 0:
                wandb.log({**wm_metrics, **ac_metrics}, step=step)
            

            # evaluation
        if step % args.eval_freq == 0:
            eval_reward = evaluate_policy(agent, args.env, eval_episodes=5, seed=args.seed + 100)
            print(f"Step {step} | Eval Reward: {eval_reward:.2f}")
            if args.wandb and wandb is not None:
                wandb.log({"eval/reward": eval_reward}, step=step)

    env.close()
    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
