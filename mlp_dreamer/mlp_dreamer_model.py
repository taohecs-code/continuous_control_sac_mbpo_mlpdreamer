import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple

nll_const = 0.5 * float(torch.log(torch.tensor(2.0 * torch.pi)))

def diag_gaussian_kl(mean_p, log_std_p, mean_q, log_std_q):
    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q)
    kl = (log_std_q - log_std_p) + (var_p + (mean_p - mean_q).pow(2)) / (2.0 * var_q) - 0.5
    return kl.sum(dim=-1, keepdim=True)

# symlog trick for sparse and high-variance rewards
# this is optinal
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)

def mlp(in_dim, out_dim, hidden_dim=256, depth=2):
    layers = []
    d = in_dim
    for _ in range(depth):
        layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

@dataclass
class DreamerConfig:
    # rssm dims
    latent_dim: int = 32
    deter_dim: int = 64
    hidden_dim: int = 256
    mlp_depth: int = 2
    horizon: int = 15

    # learning rates
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # loss weights
    kl_beta: float = 1.0
    continuation_beta: float = 1.0
    # "episode_end": same as buffer episode_end (terminated or truncated). "terminated_only": only env terminated.
    continuation_target: str = "episode_end"

    # kl stabilization
    free_nats: float = 3.0
    kl_balance: float = 0.8

    # auto kl beta
    auto_kl_beta: bool = True
    target_kl: float = 3.0
    kl_beta_lr: float = 1e-4
    kl_beta_min: float = 0.01
    kl_beta_max: float = 10.0

    # symlog scaling
    use_symlog: bool = False

    # decoder / reward: always Gaussian with learned log_std
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    target_tau: float = 0.01

    # imagination
    lambda_: float = 0.95

    imagination_backup: str = "lambda"

    # actor/critic return in imagination: "lambda_return" (Dreamer-style) or "td0" (ablation)
    return_calc_method: str = "lambda_return"

    actor_entropy_coef: float = 1e-4

    imagination_return_clip: float = None

    # grad clipping
    wm_grad_clip_norm: float = 100.0
    actor_grad_clip_norm: float = 5.0
    critic_grad_clip_norm: float = 5.0

    gamma: float = 0.99

class RSSM(nn.Module):
    def __init__(
        self,
        action_dim,
        deter_dim,
        stoch_dim,
        hidden_dim=256,
        log_std_min=-20.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Input layer: combines previous deter, previous stoch, and previous action
        self.inp = nn.Linear(deter_dim + stoch_dim + action_dim, hidden_dim)

        # RNN cell: updates the deterministic state
        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        self.prior = mlp(deter_dim, 2 * stoch_dim, hidden_dim, 1)

        self.posterior = mlp(deter_dim + stoch_dim, 2 * stoch_dim, hidden_dim, 1)

    def init_state(self, batch, device):
        deter = torch.zeros((batch, self.deter_dim), device=device)
        stoch = torch.zeros((batch, self.stoch_dim), device=device)
        return deter, stoch

    def prior_step(self, deter, stoch, action):

        x = torch.cat([deter, stoch, action], dim=-1)

        cell_inp = F.elu(self.inp(x))

        deter = self.cell(cell_inp, deter)

        stats = self.prior(deter)

        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        eps = torch.randn_like(mean)
        stoch = mean + std * eps

        return deter, stoch

class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

        depth = cfg.mlp_depth

        # 1. Encoder: obs_t -> emb_t
        self.encoder = mlp(obs_dim, cfg.latent_dim, cfg.hidden_dim, depth)

        # 2. RSSM: The dynamics engine (Prior + Posterior + RNN)
        self.rssm = RSSM(
            action_dim,
            cfg.deter_dim,
            cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )

        feat_dim = cfg.deter_dim + cfg.latent_dim

        # 3. Decoder: (deter_t, stoch_t) -> obs mean + log_std
        self.decoder = mlp(
            feat_dim,
            2 * obs_dim,
            cfg.hidden_dim,
            depth,
        )

        # 4. Reward Head: mean + log_std
        self.reward_head = mlp(
            feat_dim,
            2,
            cfg.hidden_dim,
            depth,
        )

        # 5. Continuation Head: (deter_t, stoch_t) -> p(continue)
        self.continuation_head = mlp(feat_dim, 1, cfg.hidden_dim, depth)

    def rssm_features(self, deter, stoch):
        # concat [deter, stoch] -> one feature vector
        return torch.cat([deter, stoch], dim=-1)

    def _split_mean_log_std(self, stats, out_dim):
        # split into mean | log_std, then clamp log_std
        mean, log_std = torch.split(stats, [out_dim, out_dim], dim=-1)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        return mean, log_std

    def forward_reconstruction(self, obs, actions):
        # obs (B,T,*), actions (B,T,*): unroll RSSM and train heads
        b, t, _ = obs.shape
        device = obs.device

        # init hidden at t=0
        deter, stoch = self.rssm.init_state(b, device)

        # collect per-step tensors for stack / cat
        obs_nll_list, obs_mse_list = [], []
        r_mean_list, r_log_std_list = [], []
        obs_mean_list, obs_log_std_list = [], []

        cont_logit_list, feat_list = [], []
        kl_loss_list, kl_raw_list = [], []

        # t=0 has no previous action
        prev_action = torch.zeros_like(actions[:, 0])

        # unroll in time
        for i in range(t):
            # 1. encode obs_i -> emb_i
            emb = self.encoder(obs[:, i])

            # 2. prior step: roll deter, stoch with prev_action (no current obs)
            deter, stoch = self.rssm.prior_step(deter, stoch, prev_action)

            # prior p(z|deter) for KL
            prior_stats = self.rssm.prior(deter)
            prior_mean, prior_log_std = torch.chunk(prior_stats, 2, dim=-1)
            prior_log_std = torch.clamp(prior_log_std, self.cfg.log_std_min, self.cfg.log_std_max)

            # 3. posterior q(z|deter, emb): real obs enters via emb
            post_stats = self.rssm.posterior(torch.cat([deter, emb], dim=-1))
            post_mean, post_log_std = torch.chunk(post_stats, 2, dim=-1)
            post_log_std = torch.clamp(post_log_std, self.cfg.log_std_min, self.cfg.log_std_max)
            post_std = torch.exp(post_log_std)

            # 4. KL terms
            kl_raw = diag_gaussian_kl(post_mean, post_log_std, prior_mean, prior_log_std)
            kl_raw_list.append(kl_raw)

            # KL balancing: alternate detach on post vs prior
            kl_post = diag_gaussian_kl(post_mean, post_log_std, prior_mean.detach(), prior_log_std.detach())
            kl_prior = diag_gaussian_kl(post_mean.detach(), post_log_std.detach(), prior_mean, prior_log_std)
            kl_bal = self.cfg.kl_balance * kl_post + (1.0 - self.cfg.kl_balance) * kl_prior

            # free bits: do not penalize KL below free_nats
            if self.cfg.free_nats > 0.0:
                kl_bal = F.relu(kl_bal - self.cfg.free_nats)
            kl_loss_list.append(kl_bal)

            # sample z from posterior for decoder path
            stoch = post_mean + post_std * torch.randn_like(post_mean)

            # 5. decoder + reward + continuation
            feat = self.rssm_features(deter, stoch)
            feat_list.append(feat)

            obs_mean, obs_log_std = self._split_mean_log_std(self.decoder(feat), self.obs_dim)
            r_mean, r_log_std = self._split_mean_log_std(self.reward_head(feat), 1)

            cont_logit = self.continuation_head(feat)

            obs_mean_list.append(obs_mean)
            r_mean_list.append(r_mean)
            r_log_std_list.append(r_log_std)
            obs_log_std_list.append(obs_log_std.mean(dim=-1, keepdim=True))
            cont_logit_list.append(cont_logit)

            # 6. obs: MSE for logging, Gaussian NLL for training
            obs_err = obs_mean - obs[:, i]
            obs_mse_list.append(obs_err.pow(2).mean(dim=-1, keepdim=True))

            nll = 0.5 * (obs_err.pow(2) * torch.exp(-2.0 * obs_log_std)) + obs_log_std + nll_const
            obs_nll_list.append(nll.mean(dim=-1, keepdim=True))

            prev_action = actions[:, i]

        return {
            "features": torch.stack(feat_list, dim=1),
            "rssm_features": torch.stack(feat_list, dim=1),
            "obs_hat": torch.stack(obs_mean_list, dim=1),
            "reward_pred_mean": torch.stack(r_mean_list, dim=1),
            "reward_pred_log_std": torch.stack(r_log_std_list, dim=1),
            "reward_hat": torch.stack(r_mean_list, dim=1),
            "reward_log_std": torch.stack(r_log_std_list, dim=1),
            "obs_log_std_mean": torch.stack(obs_log_std_list, dim=1),
            "continuation_pred_logit": torch.stack(cont_logit_list, dim=1),
            "continuation_logit": torch.stack(cont_logit_list, dim=1),

            "obs_nll": torch.cat(obs_nll_list, dim=1).mean(),
            "obs_mse": torch.cat(obs_mse_list, dim=1).mean(),
            "kl_loss": torch.cat(kl_loss_list, dim=1).mean(),
            "kl_raw": torch.cat(kl_raw_list, dim=1).mean(),
        }

    @torch.no_grad()
    def infer_posterior_last_state(self, obs, actions):
        # burn-in on real (obs, a) seq -> last (deter, stoch) for imagination
        b, t, _ = obs.shape
        device = obs.device

        deter, stoch = self.rssm.init_state(b, device)
        prev_action = torch.zeros_like(actions[:, 0])

        for i in range(t):
            emb = self.encoder(obs[:, i])

            deter, stoch = self.rssm.prior_step(deter, stoch, prev_action)

            # posterior from (deter, emb); no separate prior-KL here (inference only)
            post_stats = self.rssm.posterior(torch.cat([deter, emb], dim=-1))
            post_mean, post_log_std = torch.chunk(post_stats, 2, dim=-1)
            post_log_std = torch.clamp(post_log_std, self.cfg.log_std_min, self.cfg.log_std_max)

            # sample posterior z (same reparam as training path)
            stoch = post_mean + torch.exp(post_log_std) * torch.randn_like(post_mean)

            prev_action = actions[:, i]

        return deter, stoch
