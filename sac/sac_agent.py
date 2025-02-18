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
        self.eval_mode = False

        if self._config['lr_milestones'] is None:
            raise ValueError('lr_milestones argument cannot be None!\nExample: --lr_milestones=100 200 300')

        lr_milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]

        self.actor = ActorNetwork(
            input_dims=obs_dim,
            learning_rate=self._config['learning_rate'],
            action_space=self.action_space,
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device']
        )

        self.critic = CriticNetwork(
            input_dim=obs_dim,
            n_actions=4,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device']
        )

        self.critic_target = CriticNetwork(
            input_dim=obs_dim,
            n_actions=4,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            device=self._config['device']
        )

        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning:
            milestones = [int(x) for x in (self._config['alpha_milestones'][0]).split(' ')]
            self.target_entropy = -torch.tensor(4).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self._config['alpha_lr'])
            self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.alpha_optim, milestones=milestones, gamma=0.5
            )

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
        state = torch.from_numpy(obs).float().to(self.actor.device).unsqueeze(0)
        
        #state = torch.FloatTensor(obs).to(self.actor.device).unsqueeze(0) #orig
        if evaluate is False:
            action, _, _, _ = self.actor.sample(state)
        else:
            _, _, action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    # def act(self, obs):
    #     """
    #     Wrapper for _act to handle cases where obs is a tuple.
    #     """
    #     # Check if obs is a tuple
    #     if isinstance(obs, tuple):
    #         print("Warning: obs is a tuple. Extracting first element.")
    #         obs = obs[0]  # Extract the relevant part of the tuple
        
    #     return self._act(obs, evaluate=True) if self.eval_mode else self._act(obs)

    # def _act(self, obs, evaluate=False):
    #     """
    #     Perform an action based on the observation.
    #     """
    #     # Ensure `obs` is a numpy array
    #     if not isinstance(obs, np.ndarray):
    #         obs = np.array(obs, dtype=np.float32)  # Explicitly set dtype to float32
        
    #     # Handle edge case where obs has an unexpected shape
    #     if obs.ndim == 1:  # Flatten 1D arrays if necessary
    #         obs = obs[np.newaxis, :]  # Add batch dimension
        
    #     # Create the PyTorch tensor efficiently
    #     state = torch.from_numpy(obs).float().to(self.actor.device)

    #     # Sample action based on evaluate flag
    #     if not evaluate:
    #         action, _, _, _ = self.actor.sample(state)
    #     else:
    #         _, _, action, _ = self.actor.sample(state)
        
    #     return action.detach().cpu().numpy()[0]

    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.actor.lr_scheduler.step()

    def update_parameters(self, total_step):
        data = self.buffer.sample(self._config['batch_size'])

        state = torch.FloatTensor(
            np.stack(data[:, 0]),
            device=self.device
        )

        next_state = torch.FloatTensor(
            np.stack(data[:, 3]),
            device=self.device
        )

        action = torch.FloatTensor(
            np.stack(data[:, 1])[:, None],
            device=self.device
        ).squeeze(dim=1)

        reward = torch.FloatTensor(
            np.stack(data[:, 2])[:, None],
            device=self.device
        ).squeeze(dim=1)

        not_done = torch.FloatTensor(
            (~np.stack(data[:, 4])[:, None]).astype(np.int32),
            device=self.device
        ).squeeze(dim=1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.actor.sample(next_state)
            q1_next_targ, q2_next_targ = self.critic_target(next_state, next_state_action)

            min_qf_next_target = torch.min(q1_next_targ, q2_next_targ) - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self._config['gamma'] * (min_qf_next_target).squeeze()

        qf1, qf2 = self.critic(state, action)

        qf1_loss = self.critic.loss(qf1.squeeze(), next_q_value)
        qf2_loss = self.critic.loss(qf2.squeeze(), next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        qf_loss.backward()
        self.critic.optimizer.step()

        pi, log_pi, _, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean(axis=0)

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_scheduler.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if total_step % self._config['update_target_every'] == 0:
            soft_update(self.critic_target, self.critic, self._config['soft_tau'])

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item())
