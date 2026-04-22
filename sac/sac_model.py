import torch
import torch.nn as nn


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


def soft_update_target_network(source_net: nn.Module, target_net: nn.Module, tau: float):
    with torch.no_grad():
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def build_mlp_backbone(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """Shared trunk: `num_layers` blocks of Linear -> ReLU; output dim is `hidden_dim`."""
    layers: list[nn.Module] = []
    d = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
        d = hidden_dim
    return nn.Sequential(*layers)


class Actor(nn.Module):
    # SAC Actor: outputs a Gaussian policy and uses the reparameterization trick
    def __init__(
        self,
        state_dim,
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

        self.net = build_mlp_backbone(state_dim, hidden_dim, num_layers)

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, deterministic=False):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        action_scale = self.action_scale.to(device=state.device, dtype=state.dtype)
        action_bias = self.action_bias.to(device=state.device, dtype=state.dtype)

        if deterministic:
            action = tanh_gaussian_deterministic_action(mean, action_scale, action_bias)
            return action, None

        action, log_prob = tanh_gaussian_sample_and_logprob(
            mean, log_std, std, action_scale, action_bias, self.action_dim
        )
        return action, log_prob

class Critic(nn.Module):
    # Twin-Q Critic: two Q networks to mitigate overestimation bias
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1_net = nn.Sequential(
            build_mlp_backbone(in_dim, hidden_dim, num_layers),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_net = nn.Sequential(
            build_mlp_backbone(in_dim, hidden_dim, num_layers),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        # concat state and action
        x = torch.cat([state, action], dim=-1)

        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2
