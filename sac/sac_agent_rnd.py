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
from torch.func import functional_call
from memory_profiler import profile


# USING BASE SCRIPTS FROM 1. PLACE 2021 COMPETITION
# https://github.com/anticdimi/laser-hockey 

# Further Improvements: 
# AdamW: https://doi.org/10.1007/978-3-031-33374-3_26
# Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952
# Improved Entropy Tuning META-SAC: https://arxiv.org/pdf/2007.01932
# RND: https://arxiv.org/pdf/1810.12894.pdf

# @added and modified
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
    # @modified
    def __init__(self, logger, obs_dim, action_space, userconfig):
        super().__init__(
            logger=logger,
            obs_dim=obs_dim,
            action_dim=4,
            userconfig=userconfig
        )
        self.action_space = action_space
        self.device = userconfig['device']
        self.alpha = self._config.get('alpha', 0.2)
        self.automatic_entropy_tuning = self._config.get('automatic_entropy_tuning', False)
        self.meta_tuning = self._config['meta_tuning']
        self.eval_mode = False
        # inital state buffer
        self.initial_state_buffer = deque(maxlen=1000)
        self.max_episodes = self._config['max_episodes']

        # Check for prio buffer
        # Initialize replay buffer based on --per flag
        if self._config.get('per', False):  # Check if --per is enabled
            print("Using PrioritizedExperienceReplay buffer")
            self.buffer = PrioritizedExperienceReplay(
                max_size=self._config['buffer_size'],
                alpha=self._config['per_alpha'],
                beta=self._config['per_beta'],
                epsilon=1e-5
            )
        else:
            print("Using UniformExperienceReplay buffer")
            self.buffer = UniformExperienceReplay(max_size=self._config['buffer_size'])

        if self._config['lr_milestones'] is None:
            print("Using exponential decay for learning rate")
            lr_milestones = None
        else:
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
            dict_adamw=dict_adamw,
            config=self._config
        ).to(self.device) # doppelt haelt besser

        self.critic = CriticNetwork(
            input_dim=obs_dim,
            n_actions=4,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device'],
            dict_adamw=dict_adamw,
            config=self._config
        ).to(self.device) # doppelt haelt besser
 
        self.critic_target = CriticNetwork(
            input_dim=obs_dim,
            n_actions=4,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            lr_milestones=lr_milestones,
            device=self._config['device'],
            dict_adamw=dict_adamw,
            config=self._config
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

        # RND scheduler exp and linear
        self.rnd_scheduler = torch.optim.lr_scheduler.LinearLR(self.rnd_optimizer, 
                                                               start_factor=1.0, 
                                                               end_factor=0.9,#self._config['lr_factor'], 
                                                               total_iters=(self._config['max_episodes'] * 250) / 2
                                                               )

        # Intrinsic reward weight
        self.beta = userconfig['beta']
        # Added decay so less random movement in later training stages
        self.beta_start = self._config.get("beta", 1.0) 
        self.beta_end = self._config.get("beta_end", 1.0)    # Final beta after decay
        self.beta_decay = 0.999#5  # Adjust for ~2000 episodes

        # RND normalization stats
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False).to(self.device)
        self.obs_std = nn.Parameter(torch.ones(obs_dim), requires_grad=False).to(self.device)

        # Prediction Error Normalization
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0

        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning:
            if self._config['alpha_milestones'] is None:
                print("Using exponential decay for entropy tuning")
                milestones = None
            else:
                milestones = [int(x) for x in (self._config['alpha_milestones'][0]).split(' ')]
            self.target_entropy = -0.95 * torch.tensor(4).to(self.device)


            
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            
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
 
            if milestones is None:
                self.alpha_scheduler = torch.optim.lr_scheduler.LinearLR(self.alpha_optim, 
                                                               start_factor=1.0, 
                                                               end_factor=self._config['lr_factor'], 
                                                               total_iters=self._config['max_episodes'] * 250 
                                                               )
            else:
                self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.alpha_optim, milestones=milestones, gamma=0.5
                )

        # self meta entropy tuning
        if self.meta_tuning:
            if self._config['alpha_milestones'] is None:
                print("Using exponential decay for entropy tuning")
                milestones = None
            else:
                milestones = [int(x) for x in (self._config['alpha_milestones'][0]).split(' ')]
            # Change to Meta-SAC style parameterization
            self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(self.alpha), device=self.device))
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
                
            if milestones is None:
                self.alpha_scheduler = torch.optim.lr_scheduler.LinearLR(self.alpha_optim, 
                                                               start_factor=1.0, 
                                                               end_factor=self._config['lr_factor'], 
                                                               total_iters=self._config['max_episodes'] * 250 
                                                               )
            else:
                # Meta-SAC scheduler exponential decay
                self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.alpha_optim, milestones=milestones, gamma=0.5
                )
    
    def normalize(self, x):
        normalized = (x - self.obs_mean) / (self.obs_std + 1e-8)
        return torch.clamp(normalized, -8, 8)


    # added For meta tuning
    def store_initial_state(self, state):
        """Call this during training when resetting env"""
        self.initial_state_buffer.append(state.copy())
    # adeed for meta tuning
    def compute_meta_loss(self):
        """
        Compute the Meta-SAC meta loss by performing a differentiable one-step actor update
        and then evaluating the critic on the updated deterministic actions.
        """
        # Use a smaller meta batch size (default 32).
        meta_batch_size = self._config.get('meta_batch_size', 32)
        if len(self.initial_state_buffer) < meta_batch_size:
            #print("Not enough initial states in buffer for meta loss")
            return None

        # Sample a batch of initial states and normalize them.
        states = random.sample(self.initial_state_buffer, meta_batch_size)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        normalized_states = self.normalize(states)

        # --- Differentiable Actor Update ---
        # Compute the actor loss on these states.
        # Here we include a scaling factor 'meta_scale' to amplify the influence of α.
        meta_scale = self._config.get('meta_scale', 10.0)
        pi, log_pi, _, _ = self.actor.sample(normalized_states)
        q1_pi, q2_pi = self.critic(normalized_states, pi)
        min_qf_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((meta_scale * self.alpha * log_pi) - min_qf_pi).mean()

        # Compute gradients with respect to the actor parameters.
        actor_params = list(self.actor.parameters())
        grads = torch.autograd.grad(actor_loss, actor_params, create_graph=True)

        # Simulate a one-step gradient update using the actor learning rate.
        actor_lr = self._config.get('learning_rate', 3e-4)
        updated_params = [p - actor_lr * g for p, g in zip(actor_params, grads)]

        # Create an updated state_dict mapping parameter names to updated parameters.
        updated_state_dict = {}
        state_dict_keys = list(self.actor.state_dict().keys())
        for key, updated_param in zip(state_dict_keys, updated_params):
            updated_state_dict[key] = updated_param

        # --- Compute Meta Loss ---
        # Use the updated actor parameters to compute new outputs.
        updated_output = functional_call(self.actor, updated_state_dict, normalized_states)
        
        # Extract deterministic actions:
        if isinstance(updated_output, tuple):
            if len(updated_output) >= 3:
                new_actions = updated_output[2]
            else:
                new_actions = updated_output[0]
        else:
            new_actions = updated_output

        # Compute the meta loss using the critic on these updated actions.
        meta_q1, _ = self.critic(normalized_states, new_actions)
        meta_loss = -meta_q1.mean()

        return meta_loss

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

    # @modified
    def _act(self, obs, evaluate=False):
        # Ensure `obs` is a numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Create the PyTorch tensor efficiently
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_state = self.normalize(state)
        
        #state = torch.FloatTensor(obs).to(self.actor.device).unsqueeze(0) #orig
        if evaluate is False:
            action, _, _, _ = self.actor.sample(normalized_state)
        else:
            _, _, action, _ = self.actor.sample(normalized_state)
        return action.detach().cpu().numpy()[0]

    # @modified
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
        normalized_obs = self.normalize(obs)

        with torch.no_grad():
            target = self.rnd_target(normalized_obs)
        predictor = self.rnd_predictor(normalized_obs)

        # Log memory usage of RND tensors
        #logging.info(f"RND target memory: {target.element_size() * target.nelement() / 1024 ** 2:.2f} MB")
        #logging.info(f"RND predictor memory: {predictor.element_size() * predictor.nelement() / 1024 ** 2:.2f} MB")
        return torch.mean((target - predictor) ** 2, dim=1)
    
    # rnd beta update
    def update_rnd_beta(self, epsiode_counter):
        """Update beta value for intrinsic reward scaling."""
        fraction = min(epsiode_counter / (self.max_episodes * 9/10), 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        #print(f"Current beta: {self.beta}")

    # @ strongly modified, added
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
        
        if self._config.get('per', False):
            transitions = data['transitions']
            weights = data['weights']
            indices = data['indices']

            state = torch.FloatTensor(np.stack([t[0] for t in transitions])).to(self.device)
            action = torch.FloatTensor(np.stack([t[1] for t in transitions])).to(self.device)
            extrinsic_reward = torch.FloatTensor(np.stack([t[2] for t in transitions])).to(self.device)
            next_state = torch.FloatTensor(np.stack([t[3] for t in transitions])).to(self.device)
            done = torch.BoolTensor(np.stack([t[4] for t in transitions])).to(self.device)
            not_done = ~done
        else:
            state = torch.FloatTensor(np.stack([t[0] for t in data])).to(self.device)
            action = torch.FloatTensor(np.stack([t[1] for t in data])).to(self.device)
            extrinsic_reward = torch.FloatTensor(np.stack([t[2] for t in data])).to(self.device)
            next_state = torch.FloatTensor(np.stack([t[3] for t in data])).to(self.device)
            done = torch.BoolTensor(np.stack([t[4] for t in data])).to(self.device)
            not_done = ~done

        # Update normalization stats during training (for RND)
        self.obs_mean.data = 0.99 * self.obs_mean + 0.01 * next_state.mean(dim=0)
        self.obs_std.data = 0.99 * self.obs_std + 0.01 * next_state.std(dim=0)

        # Step 2: Compute intrinsic rewards using RND
        intrinsic_reward = self.compute_intrinsic_reward(next_state).detach()   # MSE between target and predictor
        # Update running stats (EMA)
        self.intrinsic_reward_mean = 0.99 * self.intrinsic_reward_mean + 0.01 * intrinsic_reward.mean()
        self.intrinsic_reward_std = 0.99 * self.intrinsic_reward_std + 0.01 * intrinsic_reward.std()
        # Normalize
        intrinsic_reward = (intrinsic_reward - self.intrinsic_reward_mean) / (self.intrinsic_reward_std + 1e-8)

        combined_reward = extrinsic_reward + self.beta * intrinsic_reward  # Total reward

        # normalize inputs
        normalized_state = self.normalize(state)
        normalized_next_state = self.normalize(next_state)

        # Step 3: Train the RND predictor network
        self.rnd_optimizer.zero_grad()  # Clear gradients
        predictor = self.rnd_predictor(normalized_next_state)  # Predictor output
        with torch.no_grad():
            target = self.rnd_target(normalized_next_state)  # Target output (fixed)
        rnd_loss = F.mse_loss(predictor, target)  # MSE loss for RND
        rnd_loss.backward()  # Backpropagate RND loss
        self.rnd_optimizer.step()  # Update RND predictor weights

        # Step 4: Update critic networks
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.actor.sample(normalized_next_state)
            q1_next_targ, q2_next_targ = self.critic_target(normalized_next_state, next_state_action)
            min_qf_next_target = torch.min(q1_next_targ, q2_next_targ) - self.alpha * next_state_log_pi
            next_q_value = combined_reward.unsqueeze(1) + not_done.unsqueeze(1) * self._config['gamma'] * min_qf_next_target

        qf1, qf2 = self.critic(normalized_state, action)

        if self._config.get('per', False):
            # Apply PER weights
            weights_tensor = torch.FloatTensor(weights).to(self.device).detach()
            qf1_loss = (weights_tensor * self.critic.loss(qf1, next_q_value)).mean()
            qf2_loss = (weights_tensor * self.critic.loss(qf2, next_q_value)).mean()
            
            # Compute TD errors using min(Q1, Q2)
            with torch.no_grad():
                min_qf = torch.min(qf1, qf2)
                td_errors = torch.abs(min_qf.detach() - next_q_value.detach()).cpu().numpy().flatten()
            
            # Update priorities
            self.buffer.update_priorities(indices.flatten(), (td_errors + self.buffer._epsilon).flatten())
        else:
            qf1_loss = self.critic.loss(qf1, next_q_value)
            qf2_loss = self.critic.loss(qf2, next_q_value)

        qf_loss = qf1_loss + qf2_loss

        # Update critic networks
        self.critic.optimizer.zero_grad()  # Clear gradients
        qf_loss.backward()  # Backpropagate critic loss
        self.critic.optimizer.step()  # Update critic weights

        # Step 5: Update the SAC actor network
        pi, log_pi, _, _ = self.actor.sample(normalized_state)  # Sample action and compute log probability
        qf1_pi, qf2_pi = self.critic(normalized_state, pi)  # Compute Q-values for the sampled action
        min_qf_pi = torch.min(qf1_pi, qf2_pi)  # Use the minimum Q-value for stability
        
        # Compute policy loss (maximize Q-value and entropy)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update actor network
        self.actor.optimizer.zero_grad()  # Clear gradients
        policy_loss.backward(retain_graph=True)  # Backpropagate policy loss
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

            self.alpha = F.softplus(self.log_alpha)
        
        elif self.meta_tuning:
            
            meta_loss = self.compute_meta_loss()

            if meta_loss is not None:  
                self.alpha_optim.zero_grad()
                meta_loss.backward()

                # Clip gradients to prevent instability
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)

                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                computed_alpha_loss = meta_loss.item()

                self.alpha = F.softplus(self.log_alpha)
            else:
                #print("Self Meta isnt getting adjusted, meta loss is none")
                computed_alpha_loss = 0.0  
        else:
            computed_alpha_loss = 0.0
            # use constant alpha from agent
            self.alpha = self._config['alpha']

        # Step 7: Soft update the target critic networks
        if total_step % self._config['update_target_every'] == 0:
            soft_update(self.critic_target, self.critic, self._config['soft_tau'])

        # Return losses for logging, if alpha loss is not computed, return 0
        return (
            qf1_loss.item(),  # Q1 loss
            qf2_loss.item(),  # Q2 loss
            policy_loss.item(),  # Policy loss
            computed_alpha_loss,   # Alpha (meta) loss
            rnd_loss.item(),  # RND loss
            self.alpha.item() if self.automatic_entropy_tuning or self.meta_tuning else self.alpha,  # Alpha value
        )
