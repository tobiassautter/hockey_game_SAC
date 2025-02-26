import sys
sys.path.insert(0, '.')
from tqdm import tqdm
from AgentLong.network import Network

import torch.nn.functional as F
import ray
import time
import numpy as np
from torchmetrics.functional import f1_score
import torch
import ray
import torch
from AgentLong.utils import get_config, get_scheduler, soft_update
import os
import AgentLong.hockey_env as h_env
from AgentLong.hockey_agent import HockeyAgent
import psutil
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import OmegaConf

@ray.remote
class SharedMemory:
    current_model = None
    training_steps = 0

    def __init__(self, config):
        self.config = config
        if "EXPERIMENT_OUTPUT_DIR" not in os.environ:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.output_dir = f"runs/{timestamp}"
        else:
            timestamp = os.environ["EXPERIMENT_RUN_ID"]
            self.output_dir = os.environ["EXPERIMENT_OUTPUT_DIR"]
        self.writer = SummaryWriter(self.output_dir)
        self.config = config
        OmegaConf.save(config, os.path.join(self.output_dir, "config.yaml"))

    def set_current_model(self, state_dict):
        self.current_model = state_dict

    def get_current_model(self):
        return self.current_model
    
    def set_training_steps(self, training_steps):
        self.training_steps = training_steps
        
    def get_training_steps(self):
        return self.training_steps
    
    def log_scalars(self, scalars, training_steps=None):
        if training_steps is None:
            training_steps = self.training_steps
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, training_steps)
    
    def save_model(self, state_dict, training_steps):
        torch.save(state_dict, f"{self.output_dir}/model_{training_steps}.pth")
    
@ray.remote(max_restarts=-1)
class SelfPlay:
    def __init__(self, config, replay_buffer, shared_memory, device, rank, exploration_epsilon):
        config = get_config(source_config=config)
        self.config = config
        self.replay_buffer = replay_buffer
        self.device = device
        self.shared_memory = shared_memory
        self.exploration_epsilon = exploration_epsilon
        self.rank = rank
        self.actions = torch.from_numpy(np.array(self.config.env.action_space)).to(self.device).float()
        
        while ray.get(self.shared_memory.get_current_model.remote()) is None:
            print("Waiting for shared memory to be initialized")
            time.sleep(1.0)

        self.network = Network(config)
        self.network.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))
        self.network.to(device=self.device)
        self.network.eval()
        self.agent = HockeyAgent(config=self.config, nets=[self.network])
        
        self.step_idx = 0
        print(f"Started self-play {rank} with exploration epsilon {exploration_epsilon}")
        self.run()
        
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e6 # Resident Set Size (RSS) in MBs

    def epsilon_greedy(self, obs, forecast):
        _, action_idx_batch = self.agent.batch_model_search(obs, forecast)
        for i in range(len(action_idx_batch)):
            if np.random.uniform(0, 1) < self.exploration_epsilon:
                action_idx = np.random.randint(0, self.actions.shape[0])
                action_idx_batch[i] = action_idx
        return self.actions[action_idx_batch].cpu(), action_idx_batch

    # the loop for the self play
    def run(self):
        # Initialize buffer
        envs = [h_env.EnvWrapper(self.config) for _ in range(self.config.ray.num_envs_per_worker)]
        obs1_list = [[] for _ in range(len(envs))]
        obs2_list = [[] for _ in range(len(envs))]
        action_indices_1_list = [[] for _ in range(len(envs))]
        action_indices_2_list = [[] for _ in range(len(envs))]
        rewards_list = [[] for _ in range(len(envs))]
        forecast1_list = [[] for _ in range(len(envs))]
        forecast2_list = [[] for _ in range(len(envs))]

        # First reset step
        for i, env in enumerate(envs):
            obs1, obs2 = env.reset()
            if self.config.env.obs_augmentation:
                obs1 = h_env.EnvWrapper.augment(obs1, self.config)
                obs2 = h_env.EnvWrapper.augment(obs2, self.config)
            obs1_list[i].append(obs1)
            obs2_list[i].append(obs2)
            if self.config.env.use_forecast:
                forecast_1 = h_env.EnvWrapper.forecast(obs1, self.config.env.forecast_step, self.config.rl.frame_skip)
                forecast_2 = h_env.EnvWrapper.forecast(obs2, self.config.env.forecast_step, self.config.rl.frame_skip)
            else:
                forecast_1 = None
                forecast_2 = None
            forecast1_list[i].append(forecast_1)
            forecast2_list[i].append(forecast_2)

        # Iterate until stopped
        game_step = 0 # for model loading. Important for shared network
        stop_signal = False # for memory usage. We have memory leak in the environment because of the forecasting.
        while len(envs) > 0:
            game_step += 1
            
            # Batch inference for both players
            last_obs1_list = [obs1[-1] for obs1 in obs1_list]
            last_obs2_list = [obs2[-1] for obs2 in obs2_list]
            last_forecast1_list = [forecast1[-1] for forecast1 in forecast1_list]
            last_forecast2_list = [forecast2[-1] for forecast2 in forecast2_list]
            batch_obs_array = np.array(last_obs1_list + last_obs2_list).reshape(-1, self.config.env.obs_dim)
            batch_forecast_array = np.array(last_forecast1_list + last_forecast2_list).reshape(batch_obs_array.shape[0], -1) if self.config.env.use_forecast else None
            action_batch, action_idx_batch = self.epsilon_greedy(batch_obs_array, batch_forecast_array)
            
            # Unpack batch into individual players
            action_1_batch = action_batch[:len(envs)]
            action_2_batch = action_batch[len(envs):]
            action_idx_1_batch = action_idx_batch[:len(envs)]
            action_idx_2_batch = action_idx_batch[len(envs):]
            
            # Store the results of environment step
            next_obs_1_list = []
            next_obs_2_list = []
            trunc_list = []
            done_list = []
            next_forecast_1_list = []
            next_forecast_2_list = []
            
            # Environment step
            for i, env in enumerate(envs):
                action1 = action_1_batch[i].cpu().numpy()
                action2 = action_2_batch[i].cpu().numpy()

                for _ in range(self.config.rl.frame_skip):
                    next_obs_1, next_obs_2, reward, done, trunc = env.step(np.hstack([action1, action2]))
                    if trunc:
                        break
                done_list.append(done)
                trunc_list.append(trunc)
                next_obs_1_list.append(next_obs_1)
                next_obs_2_list.append(next_obs_2)
                action_indices_1_list[i].append(action_idx_1_batch[i])
                action_indices_2_list[i].append(action_idx_2_batch[i])
                rewards_list[i].append(reward)
                    
                if self.config.env.obs_augmentation:
                    next_obs_1_list[i] = h_env.EnvWrapper.augment(next_obs_1_list[i], self.config)
                    next_obs_2_list[i] = h_env.EnvWrapper.augment(next_obs_2_list[i], self.config)
                if self.config.env.use_forecast:
                    next_forecast_1_list.append(h_env.EnvWrapper.forecast(next_obs_1, self.config.env.forecast_step, self.config.rl.frame_skip))
                    next_forecast_2_list.append(h_env.EnvWrapper.forecast(next_obs_2, self.config.env.forecast_step, self.config.rl.frame_skip))
                else:
                    next_forecast_1_list.append(None)
                    next_forecast_2_list.append(None)
                
            # Check if we need to reset the environment after games ended
            reset_env = []
            for i in range(len(envs)):
                if trunc_list[i]:
                    self.save_trajectory(
                        obs1_list=obs1_list[i],
                        forecast1_list=forecast1_list[i],
                        reward_list=rewards_list[i],
                        action1_list=action_indices_1_list[i],
                        action2_list=action_indices_2_list[i],
                        last_obs_1=next_obs_1_list[i],
                        last_forecast=next_forecast_1_list[i],
                        done=done_list[i]
                    )
                    if self.config.rl.use_second_player_data:
                        self.save_trajectory(
                            obs1_list=obs2_list[i],
                            forecast1_list=forecast2_list[i],
                            reward_list=[-r for r in rewards_list[i]],
                            action1_list=action_indices_2_list[i],
                            action2_list=action_indices_1_list[i],
                            last_obs_1=next_obs_2_list[i],
                            last_forecast=next_forecast_2_list[i],
                            done=done_list[i]
                        )
                    reset_env.append(i)
                else:
                    obs1_list[i].append(next_obs_1_list[i])
                    obs2_list[i].append(next_obs_2_list[i])
                    forecast1_list[i].append(next_forecast_1_list[i])
                    forecast2_list[i].append(next_forecast_2_list[i])
            
            if not stop_signal:
                for i in reset_env: # Memory leak is still okay, we can continue
                    obs1, obs2 = envs[i].reset()
                    obs1_list[i] = []
                    obs2_list[i] = []
                    action_indices_1_list[i] = []
                    action_indices_2_list[i] = []
                    rewards_list[i] = []
                    forecast1_list[i] = []
                    forecast2_list[i] = []
                    if self.config.env.obs_augmentation:
                        obs1 = h_env.EnvWrapper.augment(obs1, self.config)
                        obs2 = h_env.EnvWrapper.augment(obs2, self.config)
                    obs1_list[i].append(obs1)
                    obs2_list[i].append(obs2)
                    if self.config.env.use_forecast:
                        forecast_1 = h_env.EnvWrapper.forecast(obs1, self.config.env.forecast_step, self.config.rl.frame_skip)
                        forecast_2 = h_env.EnvWrapper.forecast(obs2, self.config.env.forecast_step, self.config.rl.frame_skip)
                    else:
                        forecast_1 = None
                        forecast_2 = None
                    forecast1_list[i].append(forecast_1)
                    forecast2_list[i].append(forecast_2)
            else: # Memory leak is too high, we need to end the worker soon
                env_to_removes = reset_env
                envs = [envs_i for i, envs_i in enumerate(envs) if i not in env_to_removes]
                obs1_list = [obs1_list_i for i, obs1_list_i in enumerate(obs1_list) if i not in env_to_removes]
                obs2_list = [obs2_list_i for i, obs2_list_i in enumerate(obs2_list) if i not in env_to_removes]
                action_indices_1_list = [action_indices_1_list_i for i, action_indices_1_list_i in enumerate(action_indices_1_list) if i not in env_to_removes]
                action_indices_2_list = [action_indices_2_list_i for i, action_indices_2_list_i in enumerate(action_indices_2_list) if i not in env_to_removes]
                rewards_list = [rewards_list_i for i, rewards_list_i in enumerate(rewards_list) if i not in env_to_removes]
                forecast1_list = [forecast1_list_i for i, forecast1_list_i in enumerate(forecast1_list) if i not in env_to_removes]
                forecast2_list = [forecast2_list_i for i, forecast2_list_i in enumerate(forecast2_list) if i not in env_to_removes]
            
            # Reload weights if needed
            if game_step % self.config.ray.shared_network_load_worker_freq == 0:
                self.network.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))
            
            stop_signal = self.get_memory_usage() > self.config.ray.max_memory
        
        print(f"Killing worker {self.rank} to avoid OOM. Don't worry about this.")
        exit()

    def save_trajectory(self, obs1_list, forecast1_list, reward_list, action1_list, action2_list, last_obs_1, last_forecast, done):
        trajectory_length = len(obs1_list)
        padding = self.config.rl.td_steps + self.config.rl.unroll_steps + 1
        discount = self.config.rl.discount

        if done:
            mc_rewards = torch.FloatTensor(reward_list)
            discounts = discount ** torch.arange(len(mc_rewards), dtype=torch.float, device=mc_rewards.device) # [1, gamma, gamma^2]
            reversed_rewards = torch.flip(mc_rewards, dims=[0])
            cumulative_returns = torch.cumsum(reversed_rewards, dim=0) # [1,...,1]. Can be simplified for binary reward. But this is good enough for now.
            mc_return = torch.flip(cumulative_returns * discounts, dims=[0]) # [gamma^n, gamma^(n-1), ..., 1]
            mc_return = torch.cat([mc_return, torch.zeros(padding, device=mc_return.device)])
            
            last_obs = torch.from_numpy(last_obs_1).float()
            observations = torch.from_numpy(np.vstack(obs1_list + padding * [last_obs])).float()
            if self.config.env.use_forecast:
                last_forecast = torch.from_numpy(last_forecast).float()
                forecast = torch.from_numpy(np.vstack(forecast1_list + padding * [last_forecast])).float()
            else:
                forecast = None
            rewards = torch.FloatTensor(reward_list + padding*[0])
            random_actions_1 = np.random.randint(0, self.actions.shape[0], padding).tolist()
            actions1 = torch.LongTensor(action1_list + random_actions_1)
            random_actions_2 = np.random.randint(0, self.actions.shape[0], padding).tolist()
            actions2 = torch.LongTensor(action2_list + random_actions_2)

            dones = torch.zeros(actions1.shape[0], dtype=torch.float)
            dones[-padding:] = 1.
        else:
            observations = torch.from_numpy(np.vstack(obs1_list)).float()
            if self.config.env.use_forecast:
                forecast = torch.from_numpy(np.vstack(forecast1_list)).float()
            else:
                forecast = None
            rewards = torch.FloatTensor(reward_list).float()
            actions1 = torch.FloatTensor(action1_list).long()
            actions2 = torch.FloatTensor(action2_list).long()
            dones = torch.zeros(observations.shape[0], dtype=torch.float)
            mc_return = torch.zeros_like(rewards)
        
        while not ray.get(self.replay_buffer.add_samples.remote(observations, forecast, rewards, actions1, actions2, dones, trajectory_length, mc_return)):
            time.sleep(0.1)

@ray.remote
class ReplayBuffer:
    def __init__(self, config, shared_memory):
        self.config = config
        self.shared_memory = shared_memory
        
        # Config variables
        self.buffer_size = self.config.rl.buffer_size
        self.padding = self.config.rl.unroll_steps + self.config.rl.td_steps + 1
        
        # Buffer
        self.observations = torch.zeros((self.buffer_size, self.config.env.obs_dim), dtype=torch.float)
        self.forecasts = torch.zeros((self.buffer_size, self.config.env.forecast_step * 2), dtype=torch.float)
        self.actions1 = torch.zeros(self.buffer_size, dtype=torch.long)
        self.actions2 = torch.zeros(self.buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float)
        self.mc_return = torch.zeros(self.buffer_size, dtype=torch.float)
        self.gradient_steps = torch.zeros(self.buffer_size, dtype=torch.float)
        
        # Indices of samples and other variables
        self.samplable_indices = torch.LongTensor([])
        self.last_sample_index = 0
        self.total_frames = 0
        self.training_steps = 0
        self.trajectory_lengths = []

    def sample(self):
        if self.avg_gradient_steps_per_frame  > self.config.optimizer.max_avg_gradient_steps_per_frame or self.samplable_indices.shape[0] < self.config.rl.n_warmup:
            #print(f"Training too fast, waiting for more samples. Avg gradient steps per frame: {self.avg_gradient_steps_per_frame}. Target is {self.config.optimizer.max_avg_gradient_steps_per_frame}")
            return None
        
        batch_size = self.config.optimizer.batch_size
        if not self.config.rl.PER:
            ind = torch.randint(0, len(self.samplable_indices), (batch_size,))
        else:
            gradient_counts = self.gradient_steps[self.samplable_indices] + 1e-6
            inv_freq = 1 / gradient_counts
            prob = inv_freq / inv_freq.sum()
            ind = torch.multinomial(prob, batch_size, replacement=len(self.samplable_indices) < batch_size)
        indices = self.samplable_indices[ind]
        self.gradient_steps[indices] += 1

        time_indices = torch.arange(self.padding)[None, :] + indices[:, None] # + 1 for transitition from obs1 to obs2. Both dynamic model and td need this +1.
        time_indices = time_indices % self.buffer_size

        observations = self.observations[time_indices]
        if self.config.env.use_forecast:
            forecast = self.forecasts[time_indices]
        else:
            forecast = None
        actions1 = self.actions1[time_indices]
        actions2 = self.actions2[time_indices]
        rewards = self.rewards[time_indices]
        dones = self.dones[time_indices]
        mc_return = self.mc_return[time_indices]
        self.training_steps += 1

        return observations, forecast, actions1, actions2, rewards, dones, mc_return

    def add_samples(self, observations, forecast, rewards, actions1, actions2, dones, trajectory_length, mc_return):
        if self.samplable_indices.shape[0] > self.config.rl.n_warmup and self.avg_gradient_steps_per_frame < self.config.optimizer.min_avg_gradient_steps_per_frame:
            #print(f"Training too slow, waiting for more samples. Avg gradient steps per frame: {self.avg_gradient_steps_per_frame}. Target is {self.config.optimizer.min_avg_gradient_steps_per_frame}")
            return False
        n_samples = observations.shape[0]

        indices = torch.arange(n_samples)
        indices = (indices + self.last_sample_index) % self.buffer_size

        self.observations[indices] = observations
        if self.config.env.use_forecast:
            self.forecasts[indices] = forecast
        self.actions1[indices] = actions1
        self.actions2[indices] = actions2
        self.rewards[indices] = rewards
        self.dones[indices] = dones
        self.mc_return[indices] = mc_return
        self.gradient_steps[indices] = 0

        for i in indices:
            if self.samplable_indices.shape[0]!=0 and i==self.samplable_indices[0]:
                self.samplable_indices = self.samplable_indices[1:]

        self.samplable_indices = torch.concat([self.samplable_indices, indices[:n_samples - self.padding + 1]])
        self.last_sample_index = (self.last_sample_index + n_samples) % self.buffer_size
        self.total_frames += trajectory_length
        self.trajectory_lengths.append(trajectory_length)
        return True

    def log_statistics(self, training_steps: int): 
        self.shared_memory.log_scalars.remote({
            "replay_buffer/total_frames": self.total_frames,
            "replay_buffer/average_gradient_steps_per_frame": self.avg_gradient_steps_per_frame,
            "replay_buffer/avg_trajectory_lengths": np.mean(self.trajectory_lengths), # Should be around 60-80, if longer, the model is properly not learning anything. Going down first to 60, than converge around 70.
            "replay_buffer/collected_trajectories": len(self.trajectory_lengths)
        }, training_steps)
    
    @property
    def avg_gradient_steps_per_frame(self):
        return (self.training_steps * self.config.optimizer.batch_size) / max(self.total_frames, 1)
        

@ray.remote(max_restarts=-1)
class Evaluator:
    def __init__(self, config, shared_memory, device):
        config = get_config(source_config=config)
        self.config = config
        self.shared_memory = shared_memory
        self.device = device
        self.run()

    def _evaluate(self, net: Network, player2: str):
        net.eval()
        agent1 = HockeyAgent(config=net.config, nets=[net])
        agent2 = h_env.BasicOpponent(weak=player2 == "weak") # Only 1 needed. Is stateless.
        envs = [h_env.EnvWrapper(self.config) for _ in range(self.config.evaluation.n_games)]
        obs1_list = []
        obs2_list = []
        winners = []
        
        for seed, env in enumerate(envs):
            obs1, obs2 = env.reset(seed=seed)
            obs1_list.append(obs1)
            obs2_list.append(obs2)
            
        while len(envs) > 0:
            forecast1_list = []
            for i, obs1 in enumerate(obs1_list):
                if self.config.env.obs_augmentation:
                    obs1_list[i] = h_env.EnvWrapper.augment(obs1, self.config)
                if self.config.env.use_forecast:
                    forecast1 = h_env.EnvWrapper.forecast(obs1_list[i], self.config.env.forecast_step, self.config.rl.frame_skip)
                    forecast1_list.append(forecast1)
            
            obs1_array = np.array(obs1_list).reshape(-1, self.config.env.obs_dim)
            obs2_array = np.array(obs2_list).reshape(-1, 18)
            forecast1_array = np.array(forecast1_list).reshape(obs1_array.shape[0], -1) if len(forecast1_list) > 0 else None
            
            action1_array, _ = agent1.batch_model_search(obs1_array, forecast1_array)
            action2_array = np.array([agent2.act(obs2) for obs2 in obs2_array])

            for i, env in enumerate(envs):
                obs1, obs2, reward, _, trunc = env.step(np.hstack([action1_array[i], action2_array[i]]))
                if trunc:
                    winners.append(reward)
                else:
                    if self.config.env.obs_augmentation:
                        obs1 = h_env.EnvWrapper.augment(obs1, self.config)
                    if self.config.env.use_forecast:
                        forecast1 = h_env.EnvWrapper.forecast(obs1, self.config.env.forecast_step, self.config.rl.frame_skip)
                        forecast1_list[i] = forecast1
                    obs1_list[i] = obs1
                    obs2_list[i] = obs2
                    
            obs1_list = [obs1 for i, obs1 in enumerate(obs1_list) if not envs[i].env.done]
            obs2_list = [obs2 for i, obs2 in enumerate(obs2_list) if not envs[i].env.done]
            envs = [env for env in envs if not env.env.done]

        winners = np.array(winners)
        return {
            f"eval/win_{player2}": (winners == 1).sum(),
            f"eval/draw_{player2}": (winners == 0).sum(),
            f"eval/loss_{player2}": (winners == -1).sum()
        }
        
    def evaluate(self):
        net = Network(self.config)
        net.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))
        net.to(device=self.device)
        net.eval()
        
        training_steps = ray.get(self.shared_memory.get_training_steps.remote())
        for player2 in ["weak", "strong"]:
            results = self._evaluate(net, player2)
            self.shared_memory.log_scalars.remote(results, training_steps)
        self.shared_memory.save_model.remote({k: v.cpu() for k, v in net.state_dict().items()}, training_steps)
        
    def run(self):
        while ray.get(self.shared_memory.get_current_model.remote()) is None:
            time.sleep(1.0)
        while ray.get(self.shared_memory.get_training_steps.remote()) < self.config.evaluation.start_evaluation:
            time.sleep(30)
            print("Evaluation can not start yet. Wait for more training steps...")
        before = time.time()
        self.evaluate()
        after = time.time()
        duration = after - before
        sleep_time = max(0, self.config.evaluation.evaluation_interval - duration)
        print(f"Finished evaluation after {duration}. Will sleep for {sleep_time} and then shutdown to avoid OOM. Will restart again.")
        time.sleep(sleep_time)
        exit()

class Trainer():
    def __init__(self, config, replay_buffer, shared_memory, device):
        self.config = config
        self.device = device
        self.replay_buffer = replay_buffer
        self.shared_memory = shared_memory
        
        while ray.get(self.shared_memory.get_current_model.remote()) is None:
            time.sleep(0.1)

        self.net_policy = Network(self.config).to(device=self.device)
        self.net_target = Network(self.config).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.net_policy.parameters(), lr=config.optimizer.lr)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.training_steps = 0
        self.actions = torch.from_numpy(np.array(self.config.env.action_space)).to(self.device).float()

    # training loop
    def train(self):
        """
        observations: [bs, time horizon + td target steps, obs_dim]
        action_indices_1: [bs, time horizon + td target steps, 1]
        action_indices_1: [bs, time horizon + td target steps, 1]
        rewards: [bs, time horizon + td target steps, 1]
        q_target: [bs, time horizon + td target steps, n_actions player 2, n_actions player 1]
        dones: [bs, time horizon + td target steps, 1]
        """
        self.net_policy.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))
        self.net_target.load_state_dict(self.net_policy.state_dict())
        self.net_policy.train()
        
        for _ in tqdm(range(self.config.rl.training_steps), mininterval=2):
            self.net_target.eval()
            while (samples := ray.get(self.replay_buffer.sample.remote())) is None: # not enough samples in the replay buffer, sample more
                time.sleep(0.01)
                continue
            
            samples = [s.to(self.device) for s in samples if s is not None]
            observations, forecast, action_indices_1, action_indices_2, rewards, dones, mc_return = samples
            actions_1 = self.actions[action_indices_1]
            actions_2 = self.actions[action_indices_2]
            bs = observations.shape[0]

            # Target network forward pass
            latent_states_target_net, values_target_net = self.net_target.initial_inference(observations.flatten(end_dim=1), forecast.flatten(end_dim=1) if forecast is not None else None)
            latent_states_target_net= latent_states_target_net.reshape(observations.shape[0], observations.shape[1], -1).detach()
            values_target_net = values_target_net.reshape(observations.shape[:2]).detach()

            # Unroll dynamic model
            predicted_current_latent_states, predicted_current_values = self.net_policy.initial_inference(observations[:, 0], forecast[:, 0] if forecast is not None else None)
            value_loss, dynamic_loss, reward_loss, reward_f1 = 0, 0, 0, 0
            for i in range(self.config.rl.unroll_steps + 1):
                predicted_next_latent_states, predicted_rewards, predicted_rewards_logits, predicted_next_values = self.net_policy.recurrent_inference(predicted_current_latent_states, actions_1[:, i], actions_2[:, i])
                
                # Value target
                lambda_ = self.config.rl.td_lambda
                mc_lambda_return = 0
                for n in range(1, self.config.rl.td_steps + 1):
                    mc_lambda_rewards = rewards[:, i:i+n].reshape(bs, -1) # [bs, n]
                    mc_lambda_discount = self.config.rl.discount ** torch.arange(n, device=self.device, dtype=mc_lambda_rewards.dtype).reshape(1, -1) # [1, n]
                    mc_lambda_return = (mc_lambda_rewards * mc_lambda_discount).sum(dim=1) # [bs]
                    td_lambda_bootstrap = (self.config.rl.discount ** n) * values_target_net[:, i+n].squeeze(-1) * (1 - dones[:, i+n].squeeze(-1)) # [bs]
                    G_n = mc_lambda_return + td_lambda_bootstrap # [bs]
                    if n < self.config.rl.td_steps:
                        balance_weight = (1 - lambda_) * (lambda_ ** (n - 1))
                    else:
                        balance_weight = lambda_ ** (n - 1)
                    mc_lambda_return += balance_weight * G_n # [bs]
                td_target = mc_lambda_return.unsqueeze(-1).clip(-1+1e-5, 1-1e-5) # [bs, 1]. Clip since can be outside of range because of bootstrap.
            
                # Compute losses
                value_loss = value_loss + F.smooth_l1_loss(predicted_current_values, td_target) / (self.config.rl.unroll_steps + 1)
                reward_loss = reward_loss +  F.cross_entropy(predicted_rewards_logits, (rewards[:, i] + 1).long()) / (self.config.rl.unroll_steps + 1)
                dynamic_loss = dynamic_loss - F.cosine_similarity(predicted_current_latent_states, latent_states_target_net[:, i]) / self.config.rl.unroll_steps* int(i > 0) # First step is only encoded latent state, from step 1 we have dynamic latent state.
                
                # Metric
                reward_f1 += f1_score(predicted_rewards.flatten() + 1, rewards[:, i].flatten() + 1, average="macro", task="multiclass", num_classes=3) / (self.config.rl.unroll_steps + 1)
                
                # Next rollout
                predicted_current_latent_states = predicted_next_latent_states
                predicted_current_values = predicted_next_values
                    
            total_loss = self.config.optimizer.value_loss_weight * value_loss + self.config.optimizer.reward_loss_weight * reward_loss + self.config.optimizer.dynamic_loss_weight * dynamic_loss 
            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()
            self.scheduler.step()

            self.training_steps += 1
            self.shared_memory.set_training_steps.remote(self.training_steps)
                
            # Soft update logic
            if self.training_steps % self.config.rl.target_network_update_freq == 0:
                if self.config.rl.soft_update_critic:
                    soft_update(target=self.net_target.state_value, source=self.net_policy.state_value, tau=self.config.rl.tau)
                    for target_name, target_module in vars(self.net_target).items():
                        if "critic" not in target_name and hasattr(target_module, "load_state_dict"):
                            target_module.load_state_dict(vars(self.net_policy)[target_name].state_dict())
                elif self.config.rl.soft_update_all:
                    soft_update(target=self.net_target, source=self.net_policy, tau=self.config.rl.tau)
                else:
                    self.net_target.load_state_dict(self.net_policy.state_dict())

            # Update the network in the shared memory
            if self.training_steps % self.config.ray.shared_network_update_freq == 0:
                self.shared_memory.set_current_model.remote({k: v.cpu() for k, v in self.net_policy.state_dict().items()})
                
            # record the losses in tensorboard once in a while
            if self.training_steps % self.config.ray.logging_freq == 0:
                # Mask out values where monte_carlo_return is 0
                episode_with_reward_mask = (mc_return != 0) & (values_target_net != 0)
                return_estimation_relative_ratio = (mc_return[episode_with_reward_mask] - values_target_net[episode_with_reward_mask]) / values_target_net[episode_with_reward_mask]  # [bs, time_steps, 1]
                return_estimation_relative_ratio = return_estimation_relative_ratio[~torch.isnan(return_estimation_relative_ratio)]
                return_estimation_ratio = mc_return[episode_with_reward_mask] / values_target_net[episode_with_reward_mask] 
                return_estimation_difference = mc_return - values_target_net
                if episode_with_reward_mask.any():
                    return_value_correlation = np.corrcoef(mc_return[episode_with_reward_mask].flatten().cpu().numpy(), values_target_net[episode_with_reward_mask].flatten().cpu().numpy())[0, 1]
                    return_value_different_sign = (mc_return[episode_with_reward_mask] * values_target_net[episode_with_reward_mask] < 0).sum().item() / episode_with_reward_mask.sum().item()
                else:
                    return_value_correlation = 0
                    return_value_different_sign = 0
                td_target_var = td_target.var().item()
                self.shared_memory.log_scalars.remote({
                    "loss/total_loss": total_loss.mean().item(),
                    "loss/value_loss": value_loss.mean().item(), # Should be very low
                    "loss/dynamic_loss": dynamic_loss.mean().item(), # Should be around -1
                    "loss/reward_loss": reward_loss.mean().item(),
                    "loss/reward_f1": reward_f1.mean().item(), # Very important for debugging! Must be around 0.95 - 1.0. The higher the better. Will be around 1 after 60k training steps. But will be > 0.9 already after a few thousands.
                    "training/lr": self.scheduler.get_last_lr()[0],
                    "debug/values_residuals_variance": (predicted_current_values - td_target).var().item() / td_target_var if td_target_var != 0 else 0, # Very important for debugging! Must be around 0.4 - 0.6. The lower the better. Optimally around 0.45 after around 40k training steps. Can be quite higher before this.
                    "debug/return_estimation_relative_ratio": return_estimation_relative_ratio.mean().item() if episode_with_reward_mask.any() else 0,
                    "debug/return_estimation_ratio": return_estimation_ratio.mean().item() if episode_with_reward_mask.any() else 0, # Doesn't have to be around 1, must if always greater than 50, we have a problem.
                    "debug/return_estimation_difference": return_estimation_difference.mean().item(), # Should be around 0 with some small variance.
                    "debug/return_value_correlation": return_value_correlation, # Very important for debugging! Must be around 0.5 - 0.7. The higher the better. Optimally around 0.6. Will converge after around 20k steps. But will be around this area already after a few thousands.
                    "debug/return_value_different_sign": return_value_different_sign,
                }, self.training_steps)
                
                def grad_norm(module):
                    return np.sqrt(sum([p.grad.norm().item()**2 for p in module.parameters() if p.grad is not None]))
                def weight_norm(module):
                    return np.sqrt(sum([p.norm().item()**2 for p in module.parameters()]))
                def feature_norm(tensor):
                    return np.sqrt((tensor**2).sum().item())
                self.shared_memory.log_scalars.remote({
                    "grad_norm/total": grad_norm(self.net_policy),
                    "grad_norm/encoder": grad_norm(self.net_policy.encoder),
                    "grad_norm/state_value": grad_norm(self.net_policy.state_value),
                    "weight_norm/total": weight_norm(self.net_policy),
                    "weight_norm/encoder": weight_norm(self.net_policy.encoder),
                    "weight_norm/state_value": weight_norm(self.net_policy.state_value),
                    "effective_lr/total": grad_norm(self.net_policy) / weight_norm(self.net_policy),
                    "effective_lr/encoder": grad_norm(self.net_policy.encoder) / weight_norm(self.net_policy.encoder),
                    "effective_lr/state_value": grad_norm(self.net_policy.state_value) / weight_norm(self.net_policy.state_value),
                })
                #for key, value in self.net_policy.intermediate_features.items():
                #    self.shared_memory.log_scalars.remote({
                #        f"feature_norm/{key}": feature_norm(value),
                #    })
                #for key, value in self.net_policy.state_value.intermediate_features.items():
                #    self.shared_memory.log_scalars.remote({
                #        f"value_feature_norm/{key}": feature_norm(value),
                #    })
                #if self.config.architecture.num_experts > 1:
                #  self.shared_memory.log_scalars.remote({
                #      f"experts/gating1_outputs_mean": self.net_policy.state_value.gaiting_weights[:, 0].mean().item(),
                #      f"experts/gating2_outputs_mean": self.net_policy.state_value.gaiting_weights[:, 1].mean().item(), 
                #      f"experts/expert_difference": (self.net_policy.state_value.expert_outputs[:, 0] - self.net_policy.state_value.expert_outputs[:, 1]).mean().item(),
                #  })
                self.shared_memory.log_scalars.remote({
                    "grad_norm/reward": grad_norm(self.net_policy.reward),
                    "grad_norm/dynamic": grad_norm(self.net_policy.dynamic),
                })
                if forecast is not None:
                    self.shared_memory.log_scalars.remote({
                        "debug/forecast_min": forecast.min().item(), # Around -9 when forecast_step = 5 after around 5k training steps. If different, we have a bug.
                        "debug/forecast_max": forecast.max().item(), # Around 9 when forecast_step = 5 after around 5k training steps. If different, we have a bug.
                        "debug/forecast_mean": forecast.mean().item(), # Around 0. If different, we have a bug.
                    }, self.training_steps)
                self.replay_buffer.log_statistics.remote(self.training_steps)

def main():
    # Load config
    config = get_config()
    
    ray.init(num_cpus=8, num_gpus=torch.cuda.device_count())

    # Central storage
    shared_memory = SharedMemory.options(namespace="main", name="SharedMemory").remote(config)
    print(f"Initialized shared memory")
    
    # Initialize shared weights
    net = Network(config)
    shared_memory.set_current_model.remote(net.state_dict())
    print(f"Initialized shared weights with {sum(p.numel() for p in net.parameters())} parameters")
    
    # Replay buffer
    replay_buffer = ReplayBuffer.options(namespace="main", name="ReplayBuffer").remote(config, shared_memory)
    print(f"Initialized replay buffer")
    
    # Start evaluators on first GPU
    evaluator = Evaluator.options(namespace="main", num_cpus=1, num_gpus=0.5).remote(config,shared_memory,torch.device(f"cuda:0"))
    print("Started evaluators")

    # Start data-collection on GPUs 1...n-1
    self_plays = []
    worker_epsilons = config.ray.epsilons
    num_workers = len(worker_epsilons)
    workers_gpus = max(torch.cuda.device_count() - 1, 0.5)
    print(f"Starting workers with exploration epsilons {worker_epsilons} on {workers_gpus} GPUs")
    
    for data_collector_rank, epsilon in enumerate(worker_epsilons):
        self_play = SelfPlay.options(namespace="main", num_cpus=1/num_workers, num_gpus=workers_gpus/num_workers).remote(config, 
                                                                                                                                     replay_buffer, 
                                                                                                                                     shared_memory, 
                                                                                                                                     torch.device("cuda:0"),
                                                                                                                                     data_collector_rank,
                                                                                                                                     epsilon)
        self_plays.append(self_play)
    print("Started self-play")
    
    # Start training on GPU 0 same as evaluator
    trainer = Trainer(config, replay_buffer, shared_memory,  torch.device(f"cuda:0"))
    trainer.train()

if __name__ == '__main__':
    main()