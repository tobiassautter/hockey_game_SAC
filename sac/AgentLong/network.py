import sys
sys.path.insert(0, '.')
import torch
from torch import nn

class Scaler(nn.Module):
    def __init__(self, dim, init=1.0, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.forward_scaler = init / scale

        self.scaler = nn.Parameter(torch.full((dim,), 1.0 * scale))

    def forward(self, x):
        return self.scaler * self.forward_scaler * x

class ResidualBlock(torch.nn.Module):
    def __init__(self, inp_out_dim, hidden_dim, num_blocks, activation_class=torch.nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.inp_out_dim = inp_out_dim
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(inp_out_dim),
            activation_class(),
            torch.nn.Linear(inp_out_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            activation_class(),
            torch.nn.Linear(hidden_dim, inp_out_dim)
        )
        self.alpha_scaler = Scaler(
            dim=inp_out_dim,
            init=1/num_blocks,
            scale=1/(hidden_dim**0.5)
        )

    def forward(self, x):
        out = self.model(x)
        return x + self.alpha_scaler(out - x)


class StateValue(torch.nn.Module):
    def __init__(self, config):
        super(StateValue, self).__init__()
        self.config = config
        hidden_dim = self.config.architecture.hidden_dim
        bottle_neck_expansion_dim = self.config.architecture.bottle_neck_expansion_dim
        activation_class = getattr(
            torch.nn, self.config.architecture.activation_function)
        self.activation_function = activation_class()
        self.num_experts = self.config.architecture.num_experts
        self.gating_net = torch.nn.Linear(hidden_dim, self.num_experts)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, int(
                    bottle_neck_expansion_dim * config.architecture.critic_expand_factor)),
                torch.nn.LayerNorm(
                    int(bottle_neck_expansion_dim * config.architecture.critic_expand_factor)),
                self.activation_function,
                torch.nn.Linear(int(bottle_neck_expansion_dim *
                                config.architecture.critic_expand_factor), 1),
                torch.nn.Tanh()
            ) for _ in range(self.num_experts)
        ])

        self._initialize_weights()

        self.gaiting_weights = None
        self.expert_outputs = None
        self.intermediate_features = {}
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(module, input, output):
            self.intermediate_features[module.full_name] = output

        for name, module in self.named_modules():
            if "gating_net" in name:
                module.full_name = f"StateValue.{name} ({module.__class__.__name__})"
                module.register_forward_hook(hook_fn)
            elif "experts" in name:
                for expert_idx, expert in enumerate(self.experts):
                    for layer_idx, layer in enumerate(expert):
                        layer.full_name = f"StateValue.experts_{expert_idx}.{layer.__class__.__name__}_{layer_idx}"
                        layer.register_forward_hook(hook_fn)

    def _initialize_weights(self):
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    if m.out_features == 1:
                        torch.nn.init.zeros_(m.weight)
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.kaiming_uniform_(
                            m.weight,
                            nonlinearity='leaky_relu' if self.config.architecture.activation_function == "LeakyReLU" else "relu"
                        )
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)
        for m in self.gating_net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, latent_state):
        gating_logits = self.gating_net(
            latent_state)  # [batch_size, num_experts]
        gating_weights = torch.softmax(
            gating_logits, dim=-1)  # [batch_size, num_experts]

        # Compute each expert’s output
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(latent_state))
        # Shape: list of [batch_size, 1] => stack => [batch_size, 1, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        # Weight each expert by gating_weights
        gating_weights = gating_weights.unsqueeze(
            1)  # [batch_size, 1, num_experts]
        output = torch.sum(expert_outputs * gating_weights,
                           dim=-1)  # [batch_size, 1]
        self.gaiting_weights = gating_weights.squeeze()
        self.expert_outputs = expert_outputs.squeeze()

        return output

class RunningStats(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer("count", torch.tensor(0.0) + 1e-4)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))

    def update(self, x):
        x = x.float()
        batch_size = x.shape[0]
        new_count = self.count + batch_size

        delta = x - self.mean
        mean_update = delta.sum(dim=0) / new_count

        self.mean += mean_update
        self.m2 += (delta * delta).sum(dim=0)
        self.count = new_count

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else torch.zeros(self.dim, device=self.m2.device)

    def normalize(self, x):
        std = self.variance.sqrt()
        std[std == 0] = 1
        return (x - self.mean) / std


class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        hidden_dim = self.config.architecture.hidden_dim
        bottle_neck_expansion_dim = self.config.architecture.bottle_neck_expansion_dim
        activation_class = getattr(
            torch.nn, self.config.architecture.activation_function)
        self.activation_function = activation_class()

        encoder_input_dim = self.config.env.obs_dim
        if self.config.env.use_forecast:
            encoder_input_dim += self.config.env.forecast_step * 2

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_input_dim, hidden_dim),
            *([ResidualBlock(
                inp_out_dim=hidden_dim,
                hidden_dim=bottle_neck_expansion_dim,
                num_blocks=self.config.architecture.encoder_num_blocks,
            )
            ] * self.config.architecture.encoder_num_blocks),
            torch.nn.LayerNorm(hidden_dim),
        )

        self.dynamic = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 2 *
                            self.config.env.action_dim, hidden_dim),
            *([ResidualBlock(
                inp_out_dim=hidden_dim,
                hidden_dim=bottle_neck_expansion_dim,
                num_blocks=self.config.architecture.dynamic_num_blocks,
            )]
                * self.config.architecture.dynamic_num_blocks),
            torch.nn.LayerNorm(hidden_dim),
        )

        self.reward = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3)
        )

        # Value net
        self.state_value = StateValue(self.config)

        # Misc
        self.obs_mean = torch.nn.Parameter(torch.FloatTensor(
            [config.env.obs_mean[:config.env.obs_dim]]), requires_grad=False)
        self.obs_std = torch.nn.Parameter(torch.FloatTensor(
            [config.env.obs_std[:config.env.obs_dim]]), requires_grad=False)
        self.forecast_mean = torch.nn.Parameter(torch.FloatTensor(
            [list(config.env.forecast_mean) * config.env.forecast_step]), requires_grad=False)
        self.forecast_std = torch.nn.Parameter(torch.FloatTensor(
            [list(config.env.forecast_std) * config.env.forecast_step]), requires_grad=False)
        self.rewards_tensor = torch.nn.Parameter(
            torch.FloatTensor([-1, 0, 1]), requires_grad=False)
        self.actions_tensor = torch.nn.Parameter(
            torch.FloatTensor(config.env.action_space), requires_grad=False)
        self.n_actions = len(self.actions_tensor)

        # Debugging
        self.intermediate_features = {}
        self.register_hooks()
        self.eval()  # Standard to avoid updating running stats accidentally
        self.run_stats = RunningStats(encoder_input_dim)

    def register_hooks(self):
        def hook_fn(module, input, output):
            self.intermediate_features[module.full_name] = output

        # Iterate over all modules
        for name, module in self.named_modules():
            if name in [
                "encoder",
                "dynamic",
                "reward",
            ]:
                # Register hooks for all submodules inside the selected modules
                for sub_name, submodule in module.named_modules():
                    full_name = (
                        f"{name}.{sub_name} ({submodule.__class__.__name__})"
                        if sub_name
                        else f"{name} ({module.__class__.__name__})"
                    )  # Handle root module
                    submodule.full_name = full_name
                    submodule.register_forward_hook(hook_fn)

    def encode(self, obs, forecast):
        """
        Args:
            obs: [bs, obs_dim]
            forecast: [bs, forecast_dim]
        Return:
            latent_state: [bs, hidden_dim]
        """
        batch_shape = obs.shape[:-1]
        obs = (obs - self.obs_mean.view(*([1] * (obs.ndim - 1)), -1)
               ) / self.obs_std.view(*([1] * (obs.ndim - 1)), -1)

        if forecast is not None:
            forecast = (forecast - self.forecast_mean.view(*([1] * (
                forecast.ndim - 1)), -1)) / self.forecast_std.view(*([1] * (forecast.ndim - 1)), -1)
            obs = torch.cat([obs, forecast], dim=-1)

        # Additional run stats beside normal standardization.
        # Seems to achieve most robust behavior.
        if self.config.architecture.running_stats_norm:
            if self.training:
                self.run_stats.update(obs)
            obs = self.run_stats.normalize(obs)

        return self.encoder(obs).reshape(*batch_shape, -1)

    def initial_inference(self, obs, forecast):
        """
        Args:
            obs: [bs, obs_dim]
        Return:
            latent_state: [bs, hidden_dim]
            state_values: [bs, 1]
        """
        latent_state = self.encode(obs, forecast)
        state_values = self.state_value(latent_state)
        return latent_state, state_values

    def recurrent_inference(self, latent_state, action_1, action_2):
        """
        Args:
            latent_state: [bs, hidden_dim]
            action_1: [bs, act_dim]
            action_2: [bs, act_dim]
        Return:
            next_latent_state: [bs, hidden_dim]
            rewards: [bs, 1]
            rewards_logits: [bs, 3]
            next_state_values: [bs, 1]
        """
        next_latent_state, rewards_logits = self.forward_dynamic(
            latent_state, action_1, action_2)
        next_state_values = self.state_value(next_latent_state)
        rewards = self.rewards_tensor[rewards_logits.argmax(
            dim=-1)].unsqueeze(1)
        return next_latent_state, rewards, rewards_logits, next_state_values

    def forward_dynamic(self, latent_state, action_1, action_2):
        """
        Args:
            latent_state: [bs, hidden_dim]
        Return:
            next_latent_state: [bs, hidden_dim]
            rewards_logits: [bs, 3]
        """
        current_state_action = torch.concat([latent_state, action_1, action_2], dim=-1)
        next_latent_state = self.dynamic(current_state_action)
        reward_logits = self.reward(torch.cat([next_latent_state, latent_state], dim=-1))
        return next_latent_state, reward_logits

    @torch.no_grad()
    def policy(self, obs, forecast):
        """
        Args:
            latent_state: [bs, obs_dim]
            forecast: [bs, forecast_dim]
        Return:
            action: [bs, action_space, action_space]
        """
        bs = obs.shape[0]
        latent_state = self.encode(obs, forecast) # [bs, hidden_dim]
        actions1 = self.actions_tensor[None, None].tile( bs, self.n_actions, 1, 1).flatten(end_dim=2) # [bs, n_actions, n_actions, act_dim]
        actions2 = self.actions_tensor[None, :, None].tile(bs, 1, self.n_actions, 1).flatten(end_dim=2) # [bs, n_actions, n_actions, act_dim]
        latent_state = latent_state[:, None, None].tile(1, self.n_actions, self.n_actions, 1).flatten(end_dim=2) # [bs, n_actions, n_actions, hidden_dim]
        _, rewards, _, next_state_values = self.recurrent_inference(latent_state, actions1, actions2)
        rewards = rewards.reshape((obs.shape[0], self.n_actions, self.n_actions))
        next_state_values = next_state_values.reshape((obs.shape[0], self.n_actions, self.n_actions))
        return rewards + self.config.rl.discount * next_state_values
