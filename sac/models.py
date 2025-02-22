import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from base.network import Feedforward

# USING BASE SCRIPTS FROM 1. PLACE 2021 COMPETITION
# https://github.com/anticdimi/laser-hockey 

#Works well with SAC and AdamW : https://doi.org/10.1007/978-3-031-33374-3_26
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) #, gain=1) 
        nn.init.constant_(m.bias, 0)

    # set last layer weights to zero
    if isinstance(m, nn.Linear) and m.out_features == 1:
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
    

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, learning_rate, device, lr_milestones, config,
                  lr_factor=0.5, loss='l2', hidden_sizes=[256, 256], dict_adamw=None):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.config = config  # Pass config to access initialization settings
        self.max_episodes = config["max_episodes"]
        self.lr_factor = lr_factor
        layer_sizes = [input_dim[0] + n_actions] + hidden_sizes + [1]

        # Q1 architecture with LayerNorm
        self.q1_layers = self._build_network(layer_sizes)
        # Q2 architecture with LayerNorm
        self.q2_layers = self._build_network(layer_sizes)

        self.apply(weights_init_)

        if device.type == 'cuda':
            self.cuda()
        
        # set optimizer to AdamW else to adam
        if dict_adamw is not None: # adamw_weight_decay adamw_eps
            self.optimizer = torch.optim.AdamW(self.parameters(), 
                                               lr=learning_rate,
                                               weight_decay=dict_adamw['weight_decay'],
                                               eps=dict_adamw['eps']
                                               )
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=0.000001)


        # if parm, use multistep or exponential scheduler
        if lr_milestones is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.lr_factor, total_iters=self.max_episodes * 250 ) 
        else:    
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )

        if loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'l1':
            self.loss = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f'Unkown loss function name: {loss}')
        
    def _build_network(self, layer_sizes):
        layers = []
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:-1]:  # Exclude input and output layers
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            prev_size = size
        # Output layer (no LayerNorm or activation)
        layers.append(nn.Linear(prev_size, layer_sizes[-1]))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        # if not self.config.architecture.critic_custom_init:
        #    return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:  # Last layer
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:  # Hidden layers
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.q1_layers(xu)
        x2 = self.q2_layers(xu)
        return x1, x2

# Gaussian policy
class ActorNetwork(Feedforward):
    def __init__(self, input_dims, learning_rate, device, lr_milestones, lr_factor=0.5, config={},
                 action_space=None, hidden_sizes=[256, 256], reparam_noise=1e-6, dict_adamw=None):
        # Initialize parent Feedforward network
        super().__init__(
            input_size=input_dims[0],
            hidden_sizes=hidden_sizes,
            output_size=1,  # Parent's output size (not directly used)
            device=device
        )

        self.config = config  # Pass config to access initialization settings
        self.max_episodes = config["max_episodes"]
        self.lr_factor = lr_factor
        self.reparam_noise = reparam_noise
        self.action_space = action_space
        n_actions = 4

        # ===== Key Change: Modify parent's layers to include LayerNorm =====
        new_layers = []
        prev_size = self.input_size
        for size in self.hidden_sizes:
            new_layers.append(nn.Linear(prev_size, size))
            new_layers.append(nn.LayerNorm(size))  # Add LayerNorm
            new_layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*new_layers)  # Override parent's plain layers

        # Output heads (unchanged)
        self.mu = nn.Linear(prev_size, n_actions)
        self.log_sigma = nn.Linear(prev_size, n_actions)

        # Rest of your init code (optimizer, action scaling, etc.)
        self.learning_rate = learning_rate
        
        # set optimizer to AdamW else to adam
        if dict_adamw is not None: # adamw_weight_decay adamw_eps
            self.optimizer = torch.optim.AdamW(self.parameters(),
                                               lr=learning_rate,
                                               weight_decay=dict_adamw['weight_decay'],
                                               eps=dict_adamw['eps']
                                               )
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=0.000001)

        # if parm, use multistep or exponential scheduler
        if lr_milestones is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.lr_factor, total_iters=self.max_episodes * 250 ) 
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=lr_factor)

        if self.action_space is not None:
            self.action_scale = torch.FloatTensor(
                (action_space.high[:n_actions] - action_space.low[:n_actions]) / 2.
            ).to(self.device)

            self.action_bias = torch.FloatTensor(
                (action_space.high[:n_actions] + action_space.low[:n_actions]) / 2.
            ).to(self.device)
        else:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)

    def forward(self, state):
        x = state
        for layer in self.layers:  # Includes LayerNorm + ReLU
            x = layer(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        return mu, log_sigma

    def sample(self, state, deterministic=False):
        mu, log_sigma = self.forward(state)
        if deterministic:
            # Compute the deterministic action (using the mean)
            deterministic_action = torch.tanh(mu) * self.action_scale + self.action_bias
            # Return a tuple with the deterministic action in the first and third positions.
            return deterministic_action, None, deterministic_action, None
        
        sigma = log_sigma.exp()
        normal = Normal(mu, sigma)

        x = normal.rsample()
        y = torch.tanh(x)

        # Reparametrization
        action = y * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x)

        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum(axis=1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return action, log_prob, mu, sigma
    

# Added from : https://arxiv.org/pdf/1810.12894
# Exploration via Random Network Distillation
class RNDNetwork(nn.Module):
    """Random Network Distillation (RND) network."""
    def __init__(self, input_dim, hidden_sizes=[256, 256], output_dim=128, device='cpu'):
        super(RNDNetwork, self).__init__()
       
        self.device = device
        layers = []
        prev_size = input_dim[0]
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            # add layernorm
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, output_dim))
        self.net = nn.Sequential(*layers).to(device)
        self.apply(weights_init_)

    def forward(self, x):
        return self.net(x)

