import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


def gaussian_nll(obs: torch.Tensor, obs_mean: torch.Tensor, obs_log_std: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Gaussian Negative Log-Likelihood (NLL).
    Math: NLL = 0.5 * ((x - mu)^2 / sigma^2) + log(sigma) + 0.5 * log(2*pi)
    """
    obs_err = obs_mean - obs
    nll_const = 0.5 * float(torch.log(torch.tensor(2.0 * torch.pi)))
    nll = 0.5 * (obs_err.pow(2) * torch.exp(-2.0 * obs_log_std)) + obs_log_std + nll_const
    return nll



class DynamicsModel(nn.Module):
    # single mlp dynamics model for rollouts
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # build mlp
        layers = []
        d = state_dim + action_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # output: state delta + reward (mean and log_std)
        self.gaussian_head = nn.Linear(hidden_dim, 2 * (state_dim + 1))
        # output: termination probability
        self.terminated_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.trunk(x)

        out = self.gaussian_head(h)
        d = self.state_dim + 1
        mu, log_std = out[..., :d], out[..., d:]
        
        # clamp for stability
        log_std = torch.clamp(log_std, -20, 2)
        terminated_logit = self.terminated_head(h)
        
        return mu, log_std, terminated_logit

    def mean_prediction(self, state, action):
        mu, _, terminated_logit = self.forward(state, action)
        delta_mu = mu[..., : self.state_dim]
        reward_mu = mu[..., self.state_dim :]
        next_state_mu = state + delta_mu
        terminated_prob = torch.sigmoid(terminated_logit)
        return next_state_mu, reward_mu, terminated_prob

    def sample_prediction(self, state, action):
        mu, log_std, terminated_logit = self.forward(state, action)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        y = mu + std * eps
        delta = y[..., : self.state_dim]
        reward = y[..., self.state_dim :]
        next_state = state + delta
        terminated_prob = torch.sigmoid(terminated_logit)
        return next_state, reward, terminated_prob

class DynamicsEnsemble(nn.Module):
    """
    Ensemble of dynamics models(class DynamicsModel) for rollouts and dynamics prediction.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        ensemble_size=7, 
        hidden_dim=256, 
        top_k_models=5,
        num_layers=2,
        ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.top_k_models = top_k_models
        
        self.models = nn.ModuleList([
            DynamicsModel(state_dim, action_dim, hidden_dim, num_layers) 
            for _ in range(ensemble_size)
        ])

        self.selected_model_indices = list(range(min(top_k_models, ensemble_size)))

    @torch.no_grad()
    def predict(self, state, action):
        """
        Predict next state and reward using a random top-k model.
        """
        b = state.shape[0]
        device = state.device

        # indexing
        selected = torch.as_tensor(self.selected_model_indices, device=device, dtype=torch.long) 
        choice = torch.randint(0, selected.numel(), (b,), device=device)
        idx = selected[choice]

        next_states = torch.empty((b, self.state_dim), device=device, dtype=state.dtype)
        rewards = torch.empty((b, 1), device=device, dtype=state.dtype)
        terminated_probs = torch.empty((b, 1), device=device, dtype=state.dtype)
        
        for i, model in enumerate(self.models):

            # use bool masking to select the ith model
            # to feed all samples that need to be predicted to the ith model at once
            mask = idx == i

            # fallback if no samples need to be predicted by the ith model
            if not torch.any(mask):
                continue
            mu, log_std, term_logit = model(state[mask], action[mask])
            std = torch.exp(log_std)
            
            y = mu + std * torch.randn_like(mu)
            
            delta = y[..., : self.state_dim]
            r = y[..., self.state_dim :]
            ns = state[mask] + delta
            term_prob = torch.sigmoid(term_logit)
            
            next_states[mask] = ns
            rewards[mask] = r
            terminated_probs[mask] = term_prob

        terminated = (terminated_probs > 0.5).to(dtype=state.dtype)
        return next_states, rewards, terminated


    @torch.no_grad()
    def selection_mse(self, state, action, next_state, reward):
        """
        Use holdout data to compute MSE for model selection.
        Return the MSE for each model.
        """
        errs = []
        for model in self.models:
            mu, _, _ = model(state, action)
            
            delta_mu = mu[..., : self.state_dim]
            r_mu = mu[..., self.state_dim :]
            ns_mu = state + delta_mu

            mse_s = F.mse_loss(ns_mu, next_state, reduction="mean")
            mse_r = F.mse_loss(r_mu, reward, reduction="mean")
            errs.append(mse_s + mse_r)

        return torch.stack(errs, dim=0)


@dataclass
class ModelTrainStats:
    nll: float
    loss: float  
    mse_next_state: float
    mse_reward: float
    avg_log_std: float
    epistemic_var_next_state: float
    epistemic_var_reward: float
    selected_model_indices: list
    terminated_bce: float = 0.0
    terminated_acc: float = 0.0
    terminated_rate: float = 0.0

def train_dynamics_ensemble(ensemble, batch, optimizer, holdout_batch=None):
    # train ensemble for one step
    state, action, reward, next_state, done = batch

    delta_target = next_state - state
    y = torch.cat([delta_target, reward], dim=-1)

    if holdout_batch is None or len(holdout_batch) < 4:
        raise ValueError("Error: holdout_batch is None or len(holdout_batch) < 4")
    s_val, a_val, r_val, ns_val = holdout_batch[:4]
    nll_sum = 0.0 
    mse_s_sum = 0.0
    mse_r_sum = 0.0
    log_std_sum = 0.0
    term_bce_sum = 0.0
    term_acc_sum = 0.0

    mu_deltas = []
    mu_rewards = []

    for model in ensemble.models:
        mu, log_std, term_logit = model(state, action)
        std = torch.exp(log_std)
        nll = gaussian_nll(y, mu, log_std)
        nll = nll.mean()
        nll_sum += nll


        # termination bce loss
        pos_weight = torch.tensor([10.0], device=done.device)
        term_bce = F.binary_cross_entropy_with_logits(term_logit, done, pos_weight=pos_weight)
        term_bce_sum += term_bce

        # monitor termination accuracy
        term_prob = torch.sigmoid(term_logit)
        term_acc = ((term_prob > 0.5).to(dtype=done.dtype) == (done > 0.5).to(dtype=done.dtype)).float().mean()
        term_acc_sum += term_acc

        delta_mu = mu[..., : ensemble.state_dim].detach()
        r_mu = mu[..., ensemble.state_dim :].detach()
        ns_mu = state + delta_mu
        
        mse_s_sum += F.mse_loss(ns_mu, next_state, reduction="mean").item()
        mse_r_sum += F.mse_loss(r_mu, reward, reduction="mean").item()
        log_std_sum += log_std.detach().mean().item()

        mu_deltas.append(delta_mu)
        mu_rewards.append(r_mu)


    # optimize
    nll_avg = nll_sum / len(ensemble.models)
    term_bce_avg = term_bce_sum / len(ensemble.models)
    term_acc_avg = term_acc_sum / len(ensemble.models)
    term_rate = float(done.mean().item())

    total_loss = nll_avg + term_bce_avg

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # metrics
    mse_s_avg = mse_s_sum / len(ensemble.models)
    mse_r_avg = mse_r_sum / len(ensemble.models)
    avg_log_std = log_std_sum / len(ensemble.models)

    # select top k models
    with torch.no_grad():
        val_errs = ensemble.selection_mse(s_val, a_val, ns_val, r_val)  
        k = min(ensemble.top_k_models, ensemble.ensemble_size)
        selected = torch.topk(val_errs, k=k, largest=False).indices.tolist()
        ensemble.selected_model_indices = selected

        # monitor
        delta_stack = torch.stack(mu_deltas, dim=0)  
        reward_stack = torch.stack(mu_rewards, dim=0)  
        epistemic_var_next = delta_stack.var(dim=0, unbiased=False).mean()
        epistemic_var_reward = reward_stack.var(dim=0, unbiased=False).mean()

    return ModelTrainStats(
        nll=float(nll_avg.item()),
        loss=float(total_loss.item()),
        mse_next_state=float(mse_s_avg),
        mse_reward=float(mse_r_avg),
        avg_log_std=float(avg_log_std),
        epistemic_var_next_state=float(epistemic_var_next.item()),
        epistemic_var_reward=float(epistemic_var_reward.item()),
        selected_model_indices=list(ensemble.selected_model_indices),
        terminated_bce=float(term_bce_avg.item()),
        terminated_acc=float(term_acc_avg.item()),
        terminated_rate=float(term_rate),
    )
