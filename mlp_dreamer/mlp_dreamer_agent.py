import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_dreamer.mlp_dreamer_model import DreamerConfig, WorldModel, symlog, symexp, mlp


def tanh_gaussian_deterministic_action(mean, action_scale, action_bias):
    return torch.tanh(mean) * action_scale + action_bias


def tanh_gaussian_sample_and_logprob(mean, log_std, std, action_scale, action_bias, action_dim):
    eps = torch.randn_like(mean)
    u = mean + std * eps
    y = torch.tanh(u)
    action = y * action_scale + action_bias
    pi_const = torch.tensor(2.0 * torch.pi, device=mean.device, dtype=mean.dtype)
    logp_u = (
        -0.5
        * (
            ((u - mean) / (std + 1e-8)).pow(2)
            + 2.0 * log_std
            + torch.log(pi_const)
        )
    ).sum(dim=-1, keepdim=True)
    log_scale = (
        torch.log(action_scale).sum()
        if action_scale.ndim > 0
        else torch.log(action_scale) * float(action_dim)
    )
    log_det = log_scale + torch.log(1.0 - y.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    log_prob = logp_u - log_det
    return action, log_prob


def gaussian_nll(obs: torch.Tensor, obs_mean: torch.Tensor, obs_log_std: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Gaussian Negative Log-Likelihood (NLL).
    Math: NLL = 0.5 * ((x - mu)^2 / sigma^2) + log(sigma) + 0.5 * log(2*pi)
    """
    obs_err = obs_mean - obs
    nll_const = 0.5 * float(torch.log(torch.tensor(2.0 * torch.pi)))
    nll = 0.5 * (obs_err.pow(2) * torch.exp(-2.0 * obs_log_std)) + obs_log_std + nll_const
    return nll

def compute_td0_return(reward_seq: torch.Tensor, value_seq: torch.Tensor, pcont_seq: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(reward_seq)
    horizon = reward_seq.shape[1]
    for i in range(horizon):
        v_next = value_seq[:, i + 1]
        discount = gamma * pcont_seq[:, i]
        returns[:, i] = reward_seq[:, i] + discount * v_next
    return returns

def compute_lambda_return(reward_seq: torch.Tensor, value_seq: torch.Tensor, pcont_seq: torch.Tensor, gamma: float, lambda_: float) -> torch.Tensor:
    returns = torch.zeros_like(reward_seq)
    horizon = reward_seq.shape[1]
    G = value_seq[:, -1]
    for i in reversed(range(horizon)):
        v_next = value_seq[:, i + 1]
        discount = gamma * pcont_seq[:, i]
        G = reward_seq[:, i] + discount * ((1.0 - lambda_) * v_next + lambda_ * G)
        returns[:, i] = G
    return returns

def soft_update_target_network(source_net: nn.Module, target_net: nn.Module, tau: float):
    """
    Polyak Averaging (Soft Update) for Target Networks.
    Math: target_weight = (1 - tau) * target_weight + tau * source_weight
    This slowly pulls the target network's weights towards the source network's weights,
    providing a stable moving target for the Actor to aim at.
    """
    with torch.no_grad():
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


class Actor(nn.Module):

    def __init__(self,
    rssm_feat_dim,
    action_dim,
    hidden_dim=256,
    num_layers=2,
    action_scale=1.0,
    action_bias=0.0,
    log_std_min=-20.0,
    log_std_max=2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # input dim should be deter + stoch + action(log,std)
        self.net = mlp(rssm_feat_dim, 2 * action_dim, hidden_dim, num_layers)
        

    def forward(self, feat, deterministic=False):
        stats = self.net(feat)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        action_scale = self.action_scale.to(device=feat.device, dtype=feat.dtype)
        action_bias = self.action_bias.to(device=feat.device, dtype=feat.dtype)

        if deterministic: # for evaluation mode
            action = tanh_gaussian_deterministic_action(mean, action_scale, action_bias)
            logp = torch.zeros((feat.shape[0], 1), device=feat.device)
            return action, logp

        action, logp = tanh_gaussian_sample_and_logprob(
            mean, log_std, std, action_scale, action_bias, self.action_dim
        )
        return action, logp

class Critic(nn.Module):
    # output a single scalar value per state
    
    def __init__(self,
    rssm_feat_dim,
    hidden_dim=256,
    num_layers=2,
    ):
    
      super().__init__()
      self.net = mlp(rssm_feat_dim, 1, hidden_dim, num_layers)
    
    def forward(self, rssm_feature):
        return self.net(rssm_feature)
    

class DreamerAgent:

    def __init__(self,
    obs_dim,
    action_dim,
    action_scale=1.0,
    action_bias=0.0,
    device=None,
    cfg=None,
    ):
        self.cfg = cfg or DreamerConfig()
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.wm = WorldModel(obs_dim, action_dim, self.cfg).to(self.device)
        rssm_feat_dim = self.cfg.deter_dim + self.cfg.latent_dim

        depth = self.cfg.mlp_depth

        self.actor = Actor(
            rssm_feat_dim,
            action_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=depth,
            action_scale=action_scale,
            action_bias=action_bias,
            log_std_min=self.cfg.log_std_min,
            log_std_max=self.cfg.log_std_max,
        ).to(self.device)
        self.critic = Critic(rssm_feat_dim, hidden_dim=self.cfg.hidden_dim, num_layers=depth).to(self.device)

        self.critic_target = Critic(rssm_feat_dim, hidden_dim=self.cfg.hidden_dim, num_layers=depth).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_target.eval()

        # optimizers
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=self.cfg.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self._kl_beta_value = self.cfg.kl_beta # auto tune kl divergence

        # short-term memory for playing the game
        self._last_feat = None
        self._deter = None
        self._stoch = None
        self._prev_action = None
        
    def reset_episode(self):
        with torch.no_grad():
            deter, stoch = self.wm.rssm.init_state(batch=1, device=self.device)
            self._deter = deter
            self._stoch = stoch
            self._prev_action = torch.zeros((1, self.actor.action_dim), device=self.device)
    
    def select_action_real_env(self, obs_np, deterministic=False):
        # transform real-env obs to tensor
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.wm.eval()
        self.actor.eval()

        with torch.no_grad():
            if self._deter is None or self._stoch is None or self._prev_action is None:
                self.reset_episode()

            emb = self.wm.encoder(obs)

            deter, stoch = self.wm.rssm.prior_step(self._deter, self._stoch, self._prev_action)
            
            post_stats = self.wm.rssm.posterior(torch.cat([deter, emb], dim=-1))
            post_mean, post_log_std = torch.chunk(post_stats, 2, dim=-1)
            post_log_std = torch.clamp(post_log_std, self.cfg.log_std_min, self.cfg.log_std_max)
            stoch = post_mean + torch.exp(post_log_std) * torch.randn_like(post_mean)

            rssm_feat = self.wm.rssm_features(deter, stoch)
            action, logp = self.actor(rssm_feat, deterministic=deterministic)
            
            lo = self.actor.action_bias - self.actor.action_scale
            hi = self.actor.action_bias + self.actor.action_scale
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action = torch.clamp(action, lo, hi)

            self._prev_action = action
            self._deter = deter
            self._stoch = stoch

        return action.squeeze(0).cpu().numpy()


    def train_world_model_from_buffer(self, obs_seq, act_seq, reward_seq, terminated_seq, truncated_or_terminated_seq):
        self.wm.train()
        out = self.wm.forward_reconstruction(obs_seq, act_seq)

        pred_reward_mean = out["reward_pred_mean"]
        pred_reward_log_std = out["reward_pred_log_std"]
        pred_continuation_logit = out["continuation_pred_logit"]
        
        obs_nll = out["obs_nll"]
        obs_mse = out["obs_mse"]

        if self.cfg.use_symlog:
            reward_target = symlog(reward_seq)
            # symlog use mse(still need to)
            reward_nll = (pred_reward_mean - reward_target).pow(2).mean()
        else:
            # use nll
            reward_nll = gaussian_nll(reward_seq, pred_reward_mean, pred_reward_log_std).mean()
        
        # monitor reward mse loss(for human eyes only, not for training)
        reward_mse = (pred_reward_mean - reward_seq).pow(2).mean()
        
        if self.cfg.continuation_target == "terminated_only":
            end_seq = terminated_seq
        else:
            # "episode_end", "truncated_or_terminated" (alias): same as buffer episode_end column
            end_seq = truncated_or_terminated_seq

        continuation_target = 1.0 - end_seq
        continuation_loss = F.binary_cross_entropy_with_logits(pred_continuation_logit, continuation_target)
        

        kl_loss = out["kl_loss"]
        kl_raw = out["kl_raw"]

        # KL weight: fixed cfg.kl_beta, or _kl_beta_value updated after each wm step (see below)
        kl_beta = self._kl_beta_value if self.cfg.auto_kl_beta else self.cfg.kl_beta

        # total loss and train
        # obs(reconstruction) loss + reward loss + continuation loss + kl divergence loss
        total_loss = obs_nll + reward_nll + self.cfg.continuation_beta * continuation_loss + kl_beta * kl_loss

        self.wm_opt.zero_grad()
        total_loss.backward()
        
        # avoid exploding gradients RNN
        if self.cfg.wm_grad_clip_norm is not None and self.cfg.wm_grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.wm_grad_clip_norm)

        self.wm_opt.step()

        # Auto KL beta: if measured KL > target, increase beta (stronger prior pull); if KL < target, decrease.
        if self.cfg.auto_kl_beta:
            kl_val = float(kl_raw.detach().item())
            self._kl_beta_value = float(
                np.clip(
                    self._kl_beta_value + self.cfg.kl_beta_lr * (kl_val - self.cfg.target_kl),
                    self.cfg.kl_beta_min,
                    self.cfg.kl_beta_max,
                )
            )

        return {
            "wm/total_loss": float(total_loss.item()),
            "wm/loss": float(total_loss.item()),
            "wm/obs_nll": float(obs_nll.item()),
            "wm/reward_nll": float(reward_nll.item()),
            "wm/continuation_bce": float(continuation_loss.item()),
            "wm/obs_mse": float(obs_mse.item()),
            "wm/reward_mse": float(reward_mse.item()),
            "wm/kl": float(kl_loss.item()),
            "wm/kl_raw": float(kl_raw.item()),
            "wm/kl_beta": float(self._kl_beta_value),
        }

    def train_actor_critic_in_dreams(self, obs_seq, act_seq):
        """
        Input for burn in:
            obs_seq: (batch_size, time_steps, obs_dim)
            act_seq: (batch_size, time_steps, action_dim)
        """
        horizon = self.cfg.horizon # dream steps length
        gamma = self.cfg.gamma # discount factor
        lambda_ = self.cfg.lambda_ # params for lambda return(TD style)

        # results and record
        # state, reward, continuation probability, action log prob
        imagined_rewards = []
        imagined_con_probs = []
        imagined_features = []
        imagined_actions_logprobs = []
        # note: we use value network here
        # so we do not need action mean
        
        # set train mode
        self.wm.train()
        self.critic.train()

        # burn in to get solid start
        # shape: (batch_size, dim)
        deter, stoch = self.wm.infer_posterior_last_state(obs_seq, act_seq)

        # important:
        # at first, freeze wm and target critic here
        # and dream rollout
        self.wm.requires_grad_(False)
        self.critic_target.requires_grad_(False)

        try:
            current_rssm_input_feature = self.wm.rssm_features(deter, stoch)
            imagined_features.append(current_rssm_input_feature)

            # actor in the dream step by step
            for t in range(horizon):
                # imagine action
                action, action_logprob = self.actor(current_rssm_input_feature, deterministic=False)
                imagined_actions_logprobs.append(action_logprob)

                # imagine next state
                next_deter, next_stoch = self.wm.rssm.prior_step(deter, stoch, action)
                next_rssm_input_feature = self.wm.rssm_features(next_deter, next_stoch)
                imagined_features.append(next_rssm_input_feature)

                # imagine reward
                imagined_reward_mean, imagined_reward_log_std = torch.chunk(self.wm.reward_head(next_rssm_input_feature), 2, dim=-1)
                imagined_rewards.append(imagined_reward_mean)

                # imagine continuation probability
                imagined_continuation_logit = self.wm.continuation_head(next_rssm_input_feature)
                imagined_con_probs.append(torch.sigmoid(imagined_continuation_logit))
                
                # update current state
                current_rssm_input_feature = next_rssm_input_feature
                deter, stoch = next_deter, next_stoch

            

            # stack by time steps
            imagined_reward_seq = torch.stack(imagined_rewards, dim=1)
            imagined_con_prob_seq = torch.stack(imagined_con_probs, dim=1)
            imagined_action_logprob_seq = torch.stack(imagined_actions_logprobs, dim=1)
            imagined_feature_seq = torch.stack(imagined_features, dim=1)

            if self.cfg.use_symlog:
                imagined_reward_seq = symexp(imagined_reward_seq)
            
            # horizon_p1: horizon + 1(initial state)
            b, horizon_p1, d = imagined_feature_seq.shape

            # critic after the dream
            # flat time
            features_flat = imagined_feature_seq.reshape(b * horizon_p1, d)

            # feed into target critic
            values_flat = self.critic_target(features_flat)

            # un-flatten back to sequence shape
            imagined_value_seq = values_flat.reshape(b, horizon_p1, 1)

            # we still need to reverse the value if used symlog
            if self.cfg.use_symlog:
                imagined_value_seq = symexp(imagined_value_seq)
            

            # calculate expected return
            if self.cfg.return_calc_method == "lambda_return":
                actor_returns = compute_lambda_return(imagined_reward_seq, imagined_value_seq, imagined_con_prob_seq, gamma, lambda_)
            elif self.cfg.return_calc_method == "td0": # simple TD(0) for ablation
                actor_returns = compute_td0_return(imagined_reward_seq, imagined_value_seq, imagined_con_prob_seq, gamma)

            # stable return (if actor returns is nan, -inf or too large or too small)
            actor_returns = torch.nan_to_num(actor_returns, nan=0.0, posinf=0.0, neginf=0.0)

            if self.cfg.imagination_return_clip and self.cfg.imagination_return_clip > 0.0:
                actor_returns = torch.clamp(actor_returns, -self.cfg.imagination_return_clip, self.cfg.imagination_return_clip)


            # use gamma (policy gradient therom)
            weights = torch.ones_like(imagined_reward_seq)
            curr_weight = torch.ones_like(imagined_reward_seq[:, 0])
            for i in range(horizon):
                weights[:, i] = curr_weight
                curr_weight = curr_weight * gamma * imagined_con_prob_seq[:, i].detach()
                

            actor_loss = -(weights * (actor_returns - self.cfg.actor_entropy_coef * imagined_action_logprob_seq)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.cfg.actor_grad_clip_norm and self.cfg.actor_grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)
            self.actor_opt.step()


        finally:
            self.wm.requires_grad_(True)
            self.critic_target.requires_grad_(True)

 

        # train critic: targets without grad through WM; critic forward must retain grad
        imagined_features_seq_detach = imagined_feature_seq.detach()
        b, horizon_p1, d = imagined_features_seq_detach.shape

        with torch.no_grad():
            target_value_seq = self.critic_target(
                imagined_features_seq_detach.reshape(b * horizon_p1, d)
            ).reshape(b, horizon_p1, 1)

            if self.cfg.use_symlog:
                target_value_seq = symexp(target_value_seq)

            if self.cfg.return_calc_method == "lambda_return":
                target_returns = compute_lambda_return(
                    imagined_reward_seq.detach(),
                    target_value_seq,
                    imagined_con_prob_seq.detach(),
                    gamma,
                    lambda_,
                )
            elif self.cfg.return_calc_method == "td0":
                target_returns = compute_td0_return(
                    imagined_reward_seq.detach(),
                    target_value_seq,
                    imagined_con_prob_seq.detach(),
                    gamma,
                )
            else:
                raise ValueError(f"Unknown return_calc_method: {self.cfg.return_calc_method!r}")

            target_returns = torch.nan_to_num(target_returns, nan=0.0, posinf=0.0, neginf=0.0)
            if self.cfg.imagination_return_clip and self.cfg.imagination_return_clip > 0.0:
                target_returns = torch.clamp(
                    target_returns,
                    -self.cfg.imagination_return_clip,
                    self.cfg.imagination_return_clip,
                )

            if self.cfg.use_symlog:
                target_returns = symlog(target_returns)

        value_pred_seq = self.critic(
            imagined_features_seq_detach[:, :-1].reshape(b * horizon_p1 - b, d)
        ).reshape(b, horizon_p1 - 1, 1)

        if target_returns.dim() == 2:
            target_returns = target_returns.unsqueeze(-1)

        value_loss = (weights * F.mse_loss(value_pred_seq, target_returns, reduction="none")).mean()

        self.critic_opt.zero_grad()
        value_loss.backward()
        if self.cfg.critic_grad_clip_norm and self.cfg.critic_grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)
        self.critic_opt.step()

        with torch.no_grad():
            soft_update_target_network(self.critic, self.critic_target, self.cfg.target_tau)

        return {
            "loss/value": float(value_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "actor/entropy_est": float((-imagined_action_logprob_seq.detach()).mean().item()),
        }


    # checkpoints management
    def get_state(self):
        return {
            "cfg": self.cfg,
            "kl_beta_value": float(self._kl_beta_value),
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state(self, state):
        if "kl_beta_value" in state:
            self._kl_beta_value = float(state["kl_beta_value"])
        if "cfg" in state:
            self.cfg = state["cfg"]
            
        self.wm.load_state_dict(state["wm"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "critic_target" in state:
            self.critic_target.load_state_dict(state["critic_target"])
        else:
            self.critic_target.load_state_dict(state["critic"])
        self.wm_opt.load_state_dict(state["wm_opt"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])

