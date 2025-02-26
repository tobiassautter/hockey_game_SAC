import sys
sys.path.insert(0, '.')
import hockey.hockey_env as h_env
import numpy as np
import gymnasium as gym
from hockey.hockey_env import BasicOpponent

class MyHockeyEnv(h_env.HockeyEnv):
  def __init__(self, keep_mode: bool=True, mode: int | str | h_env.Mode = h_env.Mode.NORMAL, verbose: bool=False):
    super().__init__(keep_mode=keep_mode, mode=mode, verbose=verbose)
    
  def set_state(self, state):
    """ function to revert the state of the environment to a previous state (observation)"""
    self.player1.position = (state[[0, 1]] + [h_env.CENTER_X, h_env.CENTER_Y]).tolist()
    self.player1.angle = state[2]
    self.player1.linearVelocity = [state[3], state[4]]
    self.player1.angularVelocity = state[5]
    self.player2.position = (state[[6, 7]] + [h_env.CENTER_X, h_env.CENTER_Y]).tolist()
    self.player2.angle = state[8]
    self.player2.linearVelocity = [state[9], state[10]]
    self.player2.angularVelocity = state[11]
    self.puck.position = (state[[12, 13]] + [h_env.CENTER_X, h_env.CENTER_Y]).tolist()
    self.puck.linearVelocity = [state[14], state[15]]
    self.player1_has_puck = state[16]
    self.player2_has_puck = state[17]

class EnvWrapper(gym.Wrapper):
    def __init__(self, config=None):
        env = MyHockeyEnv(mode=h_env.Mode.NORMAL)
        super().__init__(env)
        self.env = env
        self.config = config
        
    def reset(self, seed = None):
        if seed is None:
            seed = np.random.randint(0, 1000)#1e10)
        obs1, _ = self.env.reset(seed=seed, one_starting=(seed % 2 == 0)) # For batch evaluation
        obs2 = self.env.obs_agent_two()
        return obs1, obs2

    def step(self, action):
        obs1, reward, done, _, info = self.env.step(action)
        obs2 = self.env.obs_agent_two()
        reward = info['winner']
        trunc = done
        done = np.abs(info['winner'])
        reward = info['winner']
        return obs1, obs2, reward, done, trunc
      
    @staticmethod
    def augment(obs: np.ndarray, config) -> np.ndarray:
        obs_augmented = np.empty(config.env.obs_augmentation_dim)
        obs_augmented[: obs.shape[0]] = obs

        player1 = np.array(obs[0:2])
        player2 = np.array(obs[6:8])
        puck = np.array(obs[12:14])
        goal1 = np.array([h_env.W / 2 - 250 / h_env.SCALE, h_env.H / 2])
        goal2 = np.array([h_env.W / 2 + 250 / h_env.SCALE, h_env.H / 2])

        dist = lambda x1, x2: np.linalg.norm(x1 - x2)
        
        obs_augmented[18] = dist(player1, player2)
        obs_augmented[19] = dist(player1, goal1)
        obs_augmented[20] = dist(player1, goal2)
        obs_augmented[21] = dist(player1, puck)
        
        obs_augmented[22] = dist(player2, goal1)
        obs_augmented[23] = dist(player2, goal2)
        obs_augmented[24] = dist(player2, puck)
        
        obs_augmented[25] = dist(puck, goal1)
        obs_augmented[26] = dist(puck, goal2)
        
        return obs_augmented

    @staticmethod
    def forecast(obs, n_steps, frame_skip):
      # This could be replaced by simple linear extrapolation but this is more simple, less buggy.
      
      simulation_env = EnvWrapper()
      simulation_env.reset()
      simulation_env.env.set_state(obs)
      possession1 = obs[16] > 0
      possession2 = obs[17] > 0
      
      # Only forecast the puck. Keep players in place.
      simulation_env.env.player1.angularVelocity = 0
      simulation_env.env.player2.angularVelocity = 0
      simulation_env.env.player1.linearVelocity = (0, 0)
      simulation_env.env.player2.linearVelocity = (0, 0)

      if possession1:
        action = [0, 0, 0, 1, 0, 0, 0, 0]
      elif possession2:
        action = [0, 0, 0, 0, 0, 0, 0, 1]
      elif simulation_env.env.puck.linearVelocity[0] == 0 and simulation_env.env.puck.linearVelocity[1] == 0:
        return np.array([[obs[12], obs[13]] * n_steps]).flatten()
      else:
        action = [0, 0, 0, 0, 0, 0, 0, 0]

      trajectory = []
      for i in range(n_steps * frame_skip):
        simulation_obs, _, _, _, trunc = simulation_env.step(action)
        if trunc:
          break
        if i % frame_skip == 0:
          trajectory.append(simulation_obs[12])
          trajectory.append(simulation_obs[13])

      # Repeat the last position to get the correct length
      if len(trajectory) < n_steps * 2:
        trajectory.extend([simulation_obs[12], simulation_obs[13]] * (n_steps - len(trajectory) // 2))
      
      return np.array(trajectory).flatten()