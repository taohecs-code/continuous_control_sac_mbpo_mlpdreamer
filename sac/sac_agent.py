import torch
import torch.nn.functional as F
from dataclasses import dataclass
from sac.sac_model import Actor, Critic, soft_update_target_network

@dataclass
class SACConfig:
    # default hyperparams, usually fine as is
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = False  # auto tune entropy
    target_entropy: float = None
    alpha_loss_mode: str = "legacy"
    alpha_lr: float = None
    grad_clip_norm_actor: float = None
    grad_clip_norm_critic: float = None
    mlp_depth: int = 2
    action_scale: float = 1.0
    action_bias: float = 0.0

class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_scale=1.0,
        action_bias=0.0,
        device=None,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=None,
        cfg=None,
    ):
        # fallback to default config if none provided
        if cfg is None:
            cfg = SACConfig(
                lr=lr, gamma=gamma, tau=tau, alpha=alpha,
                auto_alpha=auto_alpha, target_entropy=target_entropy,
                action_scale=action_scale, action_bias=action_bias
            )
        self.cfg = cfg

        # pick available device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = cfg.action_scale
        self.action_bias = cfg.action_bias
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.auto_alpha = cfg.auto_alpha
        self.alpha_loss_mode = cfg.alpha_loss_mode
        self.alpha_lr = cfg.alpha_lr or self.lr
        self.grad_clip_norm_actor = cfg.grad_clip_norm_actor
        self.grad_clip_norm_critic = cfg.grad_clip_norm_critic

        # default target entropy is -dim(A)
        self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(action_dim)

        # init networks: 1 actor, 2 critics (to prevent overestimation)
        depth = cfg.mlp_depth
        hidden_dim = 256
        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim,
            num_layers=depth,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        ).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim, num_layers=depth).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=hidden_dim, num_layers=depth).to(self.device)
        
        # copy weights to target network initially
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # if auto alpha, optimize log_alpha as a parameter
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                [float(torch.log(torch.tensor(self.alpha)))],
                device=self.device, requires_grad=True
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

    def select_action(self, state, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            # add batch dim
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor(state_t, deterministic)
        # to numpy for env
        return action.cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        # sample from buffer
        batch = replay_buffer.sample(batch_size)
        return self.train_from_tensors(*batch)

    def train_from_tensors(self, state, action, reward, next_state, done):
        self.actor.train()
        self.critic.train()

        # move to device
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # compute target Q, no grad here
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            # SAC core: Q value minus entropy penalty
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = reward + self.gamma * (1 - done) * target_q

        # 1. update critic
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = 0.5 * (F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm_critic:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm_critic)
        self.critic_optimizer.step()

        # 2. update actor
        action_new, log_prob = self.actor(state)
        q1_new, q2_new = self.critic(state, action_new)
        # maximize Q and entropy
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm_actor:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm_actor)
        self.actor_optimizer.step()

        # 3. update alpha (if auto)
        alpha_loss = None
        if self.auto_alpha:
            if self.alpha_loss_mode == "standard":
                alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
            else:
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        soft_update_target_network(self.critic, self.critic_target, self.tau)

        res = {
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
        }
        if alpha_loss is not None:
            res["loss/alpha"] = alpha_loss.item()
            res["alpha"] = self.alpha
        return res

    def get_state(self):
        # save checkpoint
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cfg": self.cfg,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item() if self.log_alpha is not None else None,
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
        }

    def load_state(self, state):
        # load checkpoint
        self.state_dim = state["state_dim"]
        self.action_dim = state["action_dim"]
        
        if "cfg" in state:
            self.cfg = state["cfg"]
        
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

        if self.auto_alpha and "log_alpha" in state:
            self.log_alpha.data.fill_(state["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
            if "alpha_optimizer" in state and state["alpha_optimizer"]:
                self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
