import sys

import copy
sys.path.insert(0, '.')
sys.path.insert(1, '..')
import numpy as np
from pathlib import Path
import pickle

from base.agent import Agent
from models import *
from utils.utils import hard_update, soft_update
from base.experience_replay import UniformExperienceReplay, PrioritizedExperienceReplay

from collections import deque
import random

# USING BASE SCRIPTS FROM 1. PLACE 2021 COMPETITION
# https://github.com/anticdimi/laser-hockey 

# Further Improvements: 
# AdamW: https://doi.org/10.1007/978-3-031-33374-3_26
# Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952
# Improved Entropy Tuning META-SAC: https://arxiv.org/pdf/2007.01932
# RND: https://arxiv.org/pdf/1810.12894.pdf

class SACAgent(Agent):
    """
    The SACAgent class implements a trainable Soft Actor Critic agent, as described in: https://arxiv.org/pdf/1812.05905.pdf.

    Parameters
    ----------
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_space: ndarray
        The variable specifies the action space of environment.
    userconfig:
        The variable specifies the config settings.
    """

    def __init__(self, logger, obs_dim, action_space, userconfig):
        super().__init__(
            logger=logger,
            obs_dim=obs_dim,
            action_dim=4,
            userconfig=userconfig
        )
        self.action_space = action_space
        self.device = userconfig['device']
        self.alpha = userconfig['alpha']
        self.automatic_entropy_tuning = self._config['automatic_entropy_tuning']
        self.meta_tuning = self._config['meta_tuning']
        self.eval_mode = False
        # inital state buffer
        self.initial_state_buffer = deque(maxlen=1000)

        # Check for prio buffer
        # Initialize replay buffer based on --per flag
        if self._config.get('per', False):  # Check if --per is enabled
            print("Using PrioritizedExperienceReplay buffer")
            self.buffer = PrioritizedExperienceReplay(
                max_size=self._config['buffer_size'],  # Add buffer_size to config
                alpha=self._config['per_alpha'],
                beta=self._config['per_beta'],
                epsilon=1e-5
            )
        else:
            print("Using UniformExperienceReplay buffer")
            self.buffer = UniformExperienceReplay(max_size=self._config['buffer_size'])

        if self._config['lr_milestones'] is None:
            raise ValueError('lr_milestones argument cannot be None!\nExample: --lr_milestones=100 200 300')

        lr_milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]

        # Create dictonairy from passed AdamW config adamw_eps, adamw_weight_decay
        # and pass it to the networks
        if self._config.get('adamw', True):
            dict_adamw = {
                'eps':self._config.get('adamw_eps', 1e-6),
                'weight_decay': self._config.get('adamw_weight_decay', 1e-6)
            }
        else:
            dict_adamw = None
        
        # rint(f"Using AdamW optimizer: {dict_adamw}")

        self.actor = ActorNetwork(
            input_dims=obs_dim,
            learning_rate=self._config['learning_rate'],
            action_space=self.action_space,
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device'],
            dict_adamw=dict_adamw
        ).to(self.device) # doppelt haelt besser

        self.critic = CriticNetwork(
            input_dim=obs_dim,
            n_actions=4,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device'],
            dict_adamw=dict_adamw
        ).to(self.device) # doppelt haelt besser
 
        self.critic_target = CriticNetwork(
            input_dim=obs_dim,
            n_actions=4,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            device=self._config['device'],
            dict_adamw=dict_adamw
        ).to(self.device) # doppelt haelt besser

        # Added RND Networks structures
        self.rnd_target = RNDNetwork(
            input_dim=obs_dim,
            hidden_sizes=[256, 256],  # Match SAC hidden sizes
            output_dim=128,           # Embedding size for RND
            device=self._config['device']
        ).to(self.device).eval()  # Freeze target network

        self.rnd_predictor = RNDNetwork(
            input_dim=obs_dim,
            hidden_sizes=[256, 256],
            output_dim=128,
            device=self._config['device']
        ).to(self.device)

        # RND optimizer, check if adamw config is set, else use adam
        if dict_adamw is not None:
            self.rnd_optimizer = torch.optim.AdamW(
                self.rnd_predictor.parameters(),
                lr=userconfig['rnd_lr'],
                weight_decay=dict_adamw['weight_decay'],
                eps=dict_adamw['eps']
            )
        else:
            self.rnd_optimizer = torch.optim.Adam(
                self.rnd_predictor.parameters(),
                lr=userconfig['rnd_lr'], 
            )

        # RND scheduler
        self.rnd_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.rnd_optimizer, gamma=0.9999)

        # Intrinsic reward weight
        self.beta = userconfig['beta']
        # Added decay so less random movement in later training stages
        self.beta_start = 1.0  # Initial beta (match your config)
        self.beta_end = 0.1    # Final beta after decay
        self.beta_decay = 0.9995  # Adjust for ~2000 episodes

        # RND normalization stats
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False).to(self.device)
        self.obs_std = nn.Parameter(torch.ones(obs_dim), requires_grad=False).to(self.device)

        # Prediction Error Normalization
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0

        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning:
            milestones = [int(x) for x in (self._config['alpha_milestones'][0]).split(' ')]
            self.target_entropy = -torch.tensor(4).to(self.device)
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            
            # Check if adamw config is set, else use adam
            if dict_adamw is not None:
                self.alpha_optim = torch.optim.AdamW([self.log_alpha], 
                                                     lr=self._config['alpha_lr'],
                                                    weight_decay=dict_adamw['weight_decay'],
                                                    eps=dict_adamw['eps']
                                                    )
            else:
                self.alpha_optim = torch.optim.Adam([self.log_alpha],
                                                     lr=self._config['alpha_lr']
                                                    )
 
            
            self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.alpha_optim, milestones=milestones, gamma=0.5
            )

        # self meta entropy tuning
        if self.meta_tuning:
            # Change to Meta-SAC style parameterization
            self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            
            # Meta-SAC optimizer (use separate optimizer)
            if dict_adamw is not None:
                self.alpha_optim = torch.optim.AdamW([self.log_alpha], 
                                                     lr=self._config['alpha_lr'],
                                                    weight_decay=dict_adamw['weight_decay'],
                                                    eps=dict_adamw['eps']
                                                    )
            else:
                self.alpha_optim = torch.optim.Adam([self.log_alpha],
                                                     lr=self._config['alpha_lr']
                                                    )
                
            # Meta-SAC scheduler exponential decay
            self.alpha_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.alpha_optim, gamma=0.9999
            )

    # For meta tuning
    def store_initial_state(self, state):
        """Call this during training when resetting env"""
        self.initial_state_buffer.append(state.copy())

    # compute meta loss
    def compute_meta_loss(self):
        """Meta-SAC's novel meta loss calculation"""
        if len(self.initial_state_buffer) < self._config['batch_size']:
            return None
            
        # Sample initial states
        states_np = np.stack(self.initial_state_buffer, axis=0)  # Stack into a single array
        states = random.sample(states_np.tolist(), self._config['batch_size'])
        states = torch.FloatTensor(states, device=self.device)

        # Get deterministic policy actions
        with torch.no_grad():
            actions = self.actor.sample(states, deterministic=True)[2]

        # Compute Q-values with current critic
        q1, _ = self.critic(states, actions)
        return -q1.mean()


    @classmethod
    def clone_from(cls, agent):
        clone = cls(
            copy.deepcopy(agent.logger),
            copy.deepcopy(agent.obs_dim),
            copy.deepcopy(agent.action_space),
            copy.deepcopy(agent._config)
        )
        clone.critic.load_state_dict(agent.critic.state_dict())
        clone.critic_target.load_state_dict(agent.critic_target.state_dict())
        clone.actor.load_state_dict(agent.actor.state_dict())

        return clone

    @staticmethod
    def load_model(fpath):
        with open(Path(fpath), 'rb') as inp:
            return pickle.load(inp)

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def act(self, obs):
        # check if obs is tuple
        if isinstance(obs, tuple): # evaluater gives back tuple, quick fix TODO look into
            obs = obs[0]

        return self._act(obs, True) if self.eval_mode else self._act(obs)

    def _act(self, obs, evaluate=False):
        
        # Ensure `obs` is a numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Create the PyTorch tensor efficiently
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        #state = torch.FloatTensor(obs).to(self.actor.device).unsqueeze(0) #orig
        if evaluate is False:
            action, _, _, _ = self.actor.sample(state)
        else:
            _, _, action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.actor.lr_scheduler.step()

        # If using entropy tuning, step its scheduler
        if self.automatic_entropy_tuning and hasattr(self, 'alpha_scheduler'):
            self.alpha_scheduler.step()

        # If using Meta-Tuning, also step its scheduler
        if self.meta_tuning and hasattr(self, 'alpha_scheduler'):
            self.alpha_scheduler.step()


        # If RND predictor exists and has a scheduler, step it
        if hasattr(self, 'rnd_optimizer') and hasattr(self, 'rnd_scheduler'):
            self.rnd_scheduler.step()


    # Added rnd intrinsic reward
    def compute_intrinsic_reward(self, obs):
        """Calculate intrinsic reward as MSE between target and predictor outputs."""
        normalized_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        with torch.no_grad():
            target = self.rnd_target(normalized_obs)
        predictor = self.rnd_predictor(normalized_obs)
        return torch.mean((target - predictor) ** 2, dim=1)

    def update_parameters(self, total_step):
        """
        Perform a single update step for the SAC agent, including:
        - Sampling a batch of transitions from the replay buffer.
        - Computing intrinsic rewards using RND.
        - Training the RND predictor network.
        - Updating the SAC critic and actor networks.
        - Adjusting the entropy coefficient (if automatic entropy tuning is enabled).
        """

        # Step 1: Sample a batch of transitions from the replay buffer
        data = self.buffer.sample(self._config['batch_size'])
        
        # Extract components from the batch
        state = torch.FloatTensor(np.stack(data[:, 0])).to(device=self.device)  # Current state
        action = torch.FloatTensor(np.stack(data[:, 1])).to(device=self.device)   # Action taken
        extrinsic_reward = torch.FloatTensor(np.stack(data[:, 2])).to(device=self.device)   # Extrinsic reward
        next_state = torch.FloatTensor(np.stack(data[:, 3])).to(device=self.device)   # Next state
        not_done = torch.FloatTensor(~np.stack(data[:, 4])).to(device=self.device)   # Done flag (inverted)

        # Update normalization stats during training (for RND)
        self.obs_mean.data = 0.99 * self.obs_mean + 0.01 * next_state.mean(dim=0)
        self.obs_std.data = 0.99 * self.obs_std + 0.01 * next_state.std(dim=0)

        # Step 2: Compute intrinsic rewards using RND
        intrinsic_reward = self.compute_intrinsic_reward(next_state)  # MSE between target and predictor
        # Update running stats (EMA)
        self.intrinsic_reward_mean = 0.99 * self.intrinsic_reward_mean + 0.01 * intrinsic_reward.mean()
        self.intrinsic_reward_std = 0.99 * self.intrinsic_reward_std + 0.01 * intrinsic_reward.std()
        # Normalize
        intrinsic_reward = (intrinsic_reward - self.intrinsic_reward_mean) / (self.intrinsic_reward_std + 1e-8)

        combined_reward = extrinsic_reward + self.beta * intrinsic_reward  # Total reward

        # Decay beta for less random movement
        self.beta = max(self.beta_end, self.beta * self.beta_decay)

        # Step 3: Train the RND predictor network
        self.rnd_optimizer.zero_grad()  # Clear gradients
        predictor = self.rnd_predictor(next_state)  # Predictor output
        with torch.no_grad():
            target = self.rnd_target(next_state)  # Target output (fixed)
        rnd_loss = F.mse_loss(predictor, target)  # MSE loss for RND
        rnd_loss.backward()  # Backpropagate RND loss
        self.rnd_optimizer.step()  # Update RND predictor weights

        # Step 4: Update the SAC critic networks
        with torch.no_grad():
            # Sample next action and compute its log probability
            next_state_action, next_state_log_pi, _, _ = self.actor.sample(next_state)
            
            # Compute target Q-values using the target critic networks
            q1_next_targ, q2_next_targ = self.critic_target(next_state, next_state_action)
            
            # Use the minimum Q-value for stability (clipped double Q-learning)
            min_qf_next_target = torch.min(q1_next_targ, q2_next_targ) - self.alpha * next_state_log_pi
            
            next_q_value = combined_reward.unsqueeze(1) + not_done.unsqueeze(1) * self._config['gamma'] * min_qf_next_target
            #next_q_value = next_q_value.unsqueeze(1)  # Fix shape mismatch

        # Compute current Q-values
        qf1, qf2 = self.critic(state, action)  # Shape: [batch_size, 1]

        # Compute critic losses
        qf1_loss = self.critic.loss(qf1, next_q_value)
        qf2_loss = self.critic.loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update critic networks
        self.critic.optimizer.zero_grad()  # Clear gradients
        qf_loss.backward()  # Backpropagate critic loss
        self.critic.optimizer.step()  # Update critic weights

        # Step 5: Update the SAC actor network
        pi, log_pi, _, _ = self.actor.sample(state)  # Sample action and compute log probability
        qf1_pi, qf2_pi = self.critic(state, pi)  # Compute Q-values for the sampled action
        min_qf_pi = torch.min(qf1_pi, qf2_pi)  # Use the minimum Q-value for stability
        
        # Compute policy loss (maximize Q-value and entropy)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update actor network
        self.actor.optimizer.zero_grad()  # Clear gradients
        #policy_loss.backward()  # Backpropagate policy loss
        policy_loss.backward(retain_graph=True) # Fix for RuntimeError: Trying to backward through the graph a second time
        self.actor.optimizer.step()  # Update actor weights

        # Step 6: Adjust entropy coefficient (if automatic entropy tuning is enabled)
        if self.automatic_entropy_tuning:
            # Compute entropy loss
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            # Update entropy coefficient
            self.alpha_optim.zero_grad()  # Clear gradients
            alpha_loss.backward()  # Backpropagate entropy loss
            self.alpha_optim.step()  # Update entropy coefficient
            self.alpha = self.log_alpha.exp()  # Update alpha value
            computed_alpha_loss = alpha_loss.item()
        
        elif self.meta_tuning:
            # Use the stored meta loss if available; otherwise, default to 0.0
            computed_alpha_loss = self.last_meta_loss if hasattr(self, "last_meta_loss") else 0.0
        else:
            computed_alpha_loss = 0.0
        
        # Step 7: Soft update the target critic networks
        if total_step % self._config['update_target_every'] == 0:
            soft_update(self.critic_target, self.critic, self._config['soft_tau'])

        # Return losses for logging
        return (
            qf1_loss.item(),  # Q1 loss
            qf2_loss.item(),  # Q2 loss
            policy_loss.item(),  # Policy loss
            computed_alpha_loss,   # Alpha (meta) loss
            rnd_loss.item()  # RND loss
        )
