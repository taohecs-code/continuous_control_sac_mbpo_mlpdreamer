import argparse
import random
import numpy as np
import torch
import gymnasium as gym

try:
    import wandb
except ImportError:
    wandb = None

from mbpo.mbpo_agent import MBPOAgent, MBPOConfig, ReplayBuffer
from sac.sac_agent import SACConfig

def evaluate_policy(agent, env_name, eval_episodes, seed):
    """
    Evaluate the policy on the environment for some evaluation steps.
    """
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
    """
    CLI Parser.
    """
    parser = argparse.ArgumentParser(description="Clean MBPO Runner")
    
    # --- Macro Experiment Controls ---
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Total environment steps")
    parser.add_argument("--prefill_steps", type=int, default=5000, help="Random exploration steps before training")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluation frequency (in steps)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    
    # --- Logging ---
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="mbpo-student-version", help="WandB project name")
    
    # --- Core Algorithm Hyperparameters ---
    parser.add_argument("--horizon", type=int, default=5, help="Imagination horizon")
    parser.add_argument("--real_ratio", type=float, default=0.05, help="Ratio of real data in MBPO updates")
    parser.add_argument("--model_train_freq", type=int, default=250, help="Train model every N steps")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize wandb
    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=f"clean-mbpo-{args.env}-seed{args.seed}",
            config=vars(args),
        )
    
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Configure SAC
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    
    sac_cfg = SACConfig(
        action_scale=action_scale,
        action_bias=action_bias,
    )

    # Configure MBPO
    mbpo_cfg = MBPOConfig(
        horizon=args.horizon,
        real_ratio=args.real_ratio,
    )

    # Initialize Agent and Replay Buffer
    agent = MBPOAgent(
        state_dim=obs_dim, 
        action_dim=action_dim, 
        device=device, 
        sac_cfg=sac_cfg, 
        mbpo_cfg=mbpo_cfg
    )
    
    # This is the REAL replay buffer for environment interactions
    replay_buffer = ReplayBuffer(
        state_dim=obs_dim, 
        action_dim=action_dim, 
        max_size=int(1e6), 
        device=device, 
        holdout_ratio=0.2 # 20% data for model validation
    )

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    print(f"Starting MBPO on {args.env} with seed {args.seed} on device {device}")

    # Main training loop
    for step in range(1, args.max_steps + 1): 
        # 1. Interact with environment
        if step <= args.prefill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        episode_reward += float(reward)
        episode_steps += 1
        
        # 2. Add to real replay buffer (TimeLimit truncation is terminal for bootstrap)
        done_for_learning = float(terminated or truncated)
        episode_end = float(terminated or truncated)
        replay_buffer.add(obs, action, reward, next_obs, done_for_learning, episode_end)
        
        obs = next_obs

        # Handle episode end
        if terminated or truncated:
            episode_count += 1
            if args.wandb and wandb is not None:
                wandb.log({"train/episode_reward": episode_reward, "train/episode_steps": episode_steps}, step=step)

            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # 3. Train Networks
        if step > args.prefill_steps:
            loss_dict = {}
            
            # 3.1 Train SAC on real data
            real_stats = agent.train_policy_on_real(replay_buffer, batch_size=args.batch_size)
            loss_dict.update({f"sac_real/{k}": v for k, v in real_stats.items()})

            # 3.2 Train Dynamics Model (periodically)
            if step % args.model_train_freq == 0:
                model_stats = agent.train_model(replay_buffer, batch_size=args.batch_size)
                loss_dict.update({
                    "model/loss": model_stats.loss,
                    "model/mse_next_state": model_stats.mse_next_state,
                    "model/mse_reward": model_stats.mse_reward,
                })

            # 3.3 Train SAC on synthetic data (Model Pool)
            syn_stats = agent.train_policy_on_model_pool(replay_buffer, batch_size=args.batch_size)
            loss_dict.update({f"sac_syn/{k}": v for k, v in syn_stats.items()})

            # Log training metrics
            if args.wandb and wandb is not None and step % 100 == 0:
                wandb.log(loss_dict, step=step)
            
        # 4. Evaluation
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
