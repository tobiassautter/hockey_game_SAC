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
    
def _initialize_weights(self):
    if not self.config.architecture.critic_custom_init:
        return
    for m in self.reg_net.modules():
        if isinstance(m, nn.Linear):
            if m.out_features == 1:  # Last layer
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)
            else:  # Other layers
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu' if self.config.architecture.activation_function == "LeakyReLU" else "relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, learning_rate, device, lr_milestones,
                  lr_factor=0.5, loss='l2', hidden_sizes=[256, 256], dict_adamw=None):
        super(CriticNetwork, self).__init__()
        self.device = device
        layer_sizes = [input_dim[0] + n_actions] + hidden_sizes + [1]

        # Q1 architecture
        self.q1_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])

        # Q2 architecture
        self.q2_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])

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
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995) 
        else:    
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )

        if loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'l1':
            self.loss = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f'Unkown loss function name: {loss}')

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = xu
        for l in self.q1_layers[:-1]:
            x1 = F.relu(l(x1))
        x1 = self.q1_layers[-1](x1)

        x2 = xu
        for l in self.q2_layers[:-1]:
            x2 = F.relu(l(x2))
        x2 = self.q2_layers[-1](x2)

        return x1, x2

# Gaussian policy
class ActorNetwork(Feedforward):
    def __init__(self, input_dims, learning_rate, device, lr_milestones, lr_factor=0.5,
                 action_space=None, hidden_sizes=[256, 256], reparam_noise=1e-6, dict_adamw=None):
        super().__init__(
            input_size=input_dims[0],
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device
        )

        self.reparam_noise = reparam_noise
        self.action_space = action_space
        n_actions = 4

        self.mu = nn.Linear(hidden_sizes[-1], n_actions)
        self.log_sigma = nn.Linear(hidden_sizes[-1], n_actions)

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
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
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
        prob = state
        for layer in self.layers:
            prob = F.relu(layer(prob))

        mu = self.mu(prob)
        log_sigma = self.log_sigma(prob)
        log_sigma = torch.clamp(log_sigma, min=-20, max=10)

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
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, output_dim))
        self.net = nn.Sequential(*layers).to(device)
        self.apply(weights_init_)

    def forward(self, x):
        return self.net(x)

