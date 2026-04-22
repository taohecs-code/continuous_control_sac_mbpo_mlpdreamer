import torch
from dataclasses import dataclass
import numpy as np

from sac.sac_agent import SACAgent, SACConfig
from mbpo.mbpo_model import DynamicsEnsemble, train_dynamics_ensemble, ModelTrainStats


def _np_float_scalar(x) -> float:
    """
    Convert numpy/torch scalars or length-1 arrays to Python float.
    Done signals are from various sources: env.step() vs env.reset().
    """
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu", holdout_ratio=0.0):
        self.max_size = max_size
        self.ptr = 0  
        self.size = 0 

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
        self.episode_end = np.zeros((max_size, 1), dtype=np.float32)


        self.holdout_ratio = float(holdout_ratio)
        self._is_holdout = np.zeros(max_size, dtype=np.bool_)
        
        self.device = torch.device(device)

    def add(self, state, action, reward, next_state, done, episode_end=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = _np_float_scalar(done)
        self.episode_end[self.ptr] = _np_float_scalar(done if episode_end is None else episode_end)
        self._is_holdout[self.ptr] = (np.random.random() < self.holdout_ratio)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of sac.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(self.device),
            torch.FloatTensor(self.action[idx]).to(self.device),
            torch.FloatTensor(self.reward[idx]).to(self.device),
            torch.FloatTensor(self.next_state[idx]).to(self.device),
            torch.FloatTensor(self.done[idx]).to(self.device)
        )

    def sample_with_episode_end(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(self.device),
            torch.FloatTensor(self.action[idx]).to(self.device),
            torch.FloatTensor(self.reward[idx]).to(self.device),
            torch.FloatTensor(self.next_state[idx]).to(self.device),
            torch.FloatTensor(self.done[idx]).to(self.device),
            torch.FloatTensor(self.episode_end[idx]).to(self.device),
        )

    def sample_holdout(self, batch_size):
        holdout_idx = np.where(self._is_holdout[: self.size])[0]
        if holdout_idx.size == 0:
            return self.sample_with_episode_end(batch_size)
        idx = np.random.choice(holdout_idx, size=batch_size, replace=holdout_idx.size < batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(self.device),
            torch.FloatTensor(self.action[idx]).to(self.device),
            torch.FloatTensor(self.reward[idx]).to(self.device),
            torch.FloatTensor(self.next_state[idx]).to(self.device),
            torch.FloatTensor(self.done[idx]).to(self.device),
            torch.FloatTensor(self.episode_end[idx]).to(self.device),
        )

    def sample_train_only(self, batch_size):
        if self.holdout_ratio <= 0.0:
            return self.sample_with_episode_end(batch_size)
        train_idx = np.where(~self._is_holdout[: self.size])[0]
        if train_idx.size == 0:
            return self.sample_with_episode_end(batch_size)
        idx = np.random.choice(train_idx, size=batch_size, replace=train_idx.size < batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(self.device),
            torch.FloatTensor(self.action[idx]).to(self.device),
            torch.FloatTensor(self.reward[idx]).to(self.device),
            torch.FloatTensor(self.next_state[idx]).to(self.device),
            torch.FloatTensor(self.done[idx]).to(self.device),
            torch.FloatTensor(self.episode_end[idx]).to(self.device),
        )



@dataclass
class MBPOConfig:
    # model hyperparams
    horizon: int = 5
    model_ensemble_size: int = 7
    model_top_k: int = 5
    model_hidden_dim: int = 256
    model_num_layers: int = 2
    model_lr: float = 3e-4
    model_weight_decay: float = 1e-4

    # training freq
    model_train_steps_per_env_step: int = 1
    synthetic_updates_per_env_step: int = 1
    terminal_target: str = "episode_end"
    real_ratio: float = 0.0

    # model pool (always used for synthetic SAC updates)
    model_pool_size: int = 200000
    model_pool_min_size: int = 256

class MBPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device=None,
        sac_cfg=None,
        mbpo_cfg=None,
    ):
        if sac_cfg is None:
            sac_cfg = SACConfig()
        
        # mbpo uses sac as the base policy
        self.policy = SACAgent(state_dim, action_dim, device=device, cfg=sac_cfg)
        self.cfg = mbpo_cfg or MBPOConfig()

        # init dynamics model
        self.model = DynamicsEnsemble(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=self.cfg.model_ensemble_size,
            hidden_dim=self.cfg.model_hidden_dim,
            top_k_models=self.cfg.model_top_k,
            num_layers=self.cfg.model_num_layers,
        ).to(self.policy.device)

        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.model_lr,
            weight_decay=self.cfg.model_weight_decay,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model_pool = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=max(1, self.cfg.model_pool_size),
            device=self.policy.device,
        )

    def select_action(self, state, deterministic=False):
        return self.policy.select_action(state, deterministic=deterministic)

    def train_policy_on_real(self, replay_buffer, batch_size):
        # standard sac update on real data
        return self.policy.train(replay_buffer, batch_size=batch_size)

    def train_model(self, replay_buffer, batch_size):
        # for safety
        # can train model for multiple steps per env step
        n_steps = max(1, self.cfg.model_train_steps_per_env_step)

        # Logging Metrics
        sum_loss = 0.0
        sum_nll = 0.0
        sum_terminated_bce = 0.0

        sum_mse_next_state = 0.0
        sum_mse_reward = 0.0
        sum_avg_log_std = 0.0
        sum_epistemic_var_next_state = 0.0
        sum_epistemic_var_reward = 0.0
        sum_terminated_acc = 0.0
        sum_terminated_rate = 0.0

        h = replay_buffer.sample_holdout(batch_size)
        holdout_batch = tuple(t.to(self.policy.device) for t in h[:4])

        for _ in range(n_steps):
            batch = replay_buffer.sample_train_only(batch_size)
            batch = tuple(t.to(self.policy.device) for t in batch)
            state, action, reward, next_state, done_for_learning, episode_end = batch

            done_target = episode_end if self.cfg.terminal_target == "episode_end" else done_for_learning
            train_batch = (state, action, reward, next_state, done_target)

            step_stats = train_dynamics_ensemble(
                self.model, train_batch, self.model_optimizer, holdout_batch=holdout_batch
            )

            sum_loss += step_stats.loss
            sum_nll += step_stats.nll
            sum_terminated_bce += step_stats.terminated_bce

            sum_mse_next_state += step_stats.mse_next_state
            sum_mse_reward += step_stats.mse_reward
            sum_avg_log_std += step_stats.avg_log_std
            sum_epistemic_var_next_state += step_stats.epistemic_var_next_state
            sum_epistemic_var_reward += step_stats.epistemic_var_reward
            sum_terminated_acc += step_stats.terminated_acc
            sum_terminated_rate += step_stats.terminated_rate       

        inv = 1.0 / n_steps

        return ModelTrainStats(
            loss=sum_loss * inv,
            nll=sum_nll * inv,
            mse_next_state=sum_mse_next_state * inv,
            mse_reward=sum_mse_reward * inv,
            avg_log_std=sum_avg_log_std * inv,
            epistemic_var_next_state=sum_epistemic_var_next_state * inv,
            epistemic_var_reward=sum_epistemic_var_reward * inv,
            terminated_bce=sum_terminated_bce * inv,
            terminated_acc=sum_terminated_acc * inv,
            terminated_rate=sum_terminated_rate * inv,
            selected_model_indices=step_stats.selected_model_indices
        )

    @torch.no_grad()
    def rollout(self, start_states, horizon):
        """
        Imagined rollouts: chain next state through the learned model.
        `predict` already returns binary termination; use it directly for done flags.
        """
        device = self.policy.device
        s = start_states.to(device)

        batch_states, batch_actions, batch_rewards = [], [], []
        batch_next_states, batch_dones = [], []

        for _ in range(horizon):
            action_t, _ = self.policy.actor(s, deterministic=False)

            ns, r, done_t = self.model.predict(s, action_t)
            done_t = done_t.to(dtype=s.dtype)

            batch_states.append(s)
            batch_actions.append(action_t)
            batch_rewards.append(r)
            batch_next_states.append(ns)
            batch_dones.append(done_t)
            s = ns

        return (
            torch.cat(batch_states, dim=0),
            torch.cat(batch_actions, dim=0),
            torch.cat(batch_rewards, dim=0),
            torch.cat(batch_next_states, dim=0),
            torch.cat(batch_dones, dim=0),
        )


    def train_policy_on_model_pool(self, replay_buffer, batch_size):
        n_updates = max(1, self.cfg.synthetic_updates_per_env_step)

        s0, _, _, _, _ = replay_buffer.sample(batch_size)
        s0 = s0.to(self.policy.device)

        syn_batch = self.rollout(s0, horizon=self.cfg.horizon)
        s, a, r, ns, d = syn_batch

        s_np = s.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        ns_np = ns.detach().cpu().numpy()
        d_np = d.detach().cpu().numpy()

        for i in range(s_np.shape[0]):
            self.model_pool.add(
                s_np[i], a_np[i], r_np[i], ns_np[i], d_np[i], episode_end=d_np[i]
            )

        pool_ready = self.model_pool.size >= max(1, self.cfg.model_pool_min_size)

        def synthetic_sample(n: int):
            """Prefer the replay pool once filled; otherwise train on this step's rollout."""
            if n <= 0:
                return None
            if pool_ready:
                return self.model_pool.sample(n)
            total = s.shape[0]
            if total == 0:
                return None
            idx = np.random.choice(total, size=n, replace=(n > total))
            return (s[idx], a[idx], r[idx], ns[idx], d[idx])

        last_stats = {}

        for _ in range(n_updates):
            syn_batch_size = int(batch_size * (1.0 - self.cfg.real_ratio))
            real_batch_size = batch_size - syn_batch_size

            # real_ratio=1 => syn_batch_size==0: no synthetic batch; one SAC step on real transitions only
            if syn_batch_size == 0:
                if real_batch_size > 0 and replay_buffer.size >= real_batch_size:
                    real_train_batch = replay_buffer.sample(real_batch_size)
                    real_train_batch = tuple(t.to(self.policy.device) for t in real_train_batch)
                    last_stats = self.policy.train_from_tensors(*real_train_batch)
                continue

            syn_train_batch = synthetic_sample(syn_batch_size)
            if syn_train_batch is None:
                continue
            syn_train_batch = tuple(t.to(self.policy.device) for t in syn_train_batch)

            if real_batch_size > 0 and replay_buffer.size >= real_batch_size:
                real_train_batch = replay_buffer.sample(real_batch_size)
                real_train_batch = tuple(t.to(self.policy.device) for t in real_train_batch)
                mixed_batch = tuple(
                    torch.cat([syn_item, real_item], dim=0)
                    for syn_item, real_item in zip(syn_train_batch, real_train_batch[:5])
                )
                last_stats = self.policy.train_from_tensors(*mixed_batch)
            else:
                syn_full = synthetic_sample(batch_size)
                if syn_full is None:
                    continue
                syn_full = tuple(t.to(self.policy.device) for t in syn_full)
                last_stats = self.policy.train_from_tensors(*syn_full)

        return last_stats


    def get_state(self):
        # save checkpoint
        return {
            "mbpo_cfg": self.cfg,
            "policy": self.policy.get_state(),
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
        }

    def load_state(self, state):
        # load checkpoint
        self.policy.load_state(state["policy"])
        self.model.load_state_dict(state["model"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])
        if "mbpo_cfg" in state:
            self.cfg = state["mbpo_cfg"]


        