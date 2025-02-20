import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import shutil
from tabulate import tabulate
import random
from copy import deepcopy
from hockey.hockey_env import CENTER_X, CENTER_Y, SCALE, W, H, HockeyEnv, Mode, GOAL_SIZE
import os
from scipy.spatial import distance as euclidean_distance
import gymnasium as gym
# class EnvWrapper(gym.Wrapper):
#     def _init_(self, config=None, random_init=False):
#         env = HockeyEnv(mode=Mode.NORMAL)
#         super()._init_(env)
#         self.env = env
#         self.config = config
        
#     def reset(self, seed = None):
#         if seed is None:
#             seed = np.random.randint(0, 1e10)
#         obs1, _ = self.env.reset(seed=seed)
#         obs2 = self.env.obs_agent_two()
#         return obs1, obs2

#     def step(self, action):
#         obs1, reward, done, _, info = self.env.step(action)
#         obs2 = self.env.obs_agent_two()
#         reward = info['winner']
#         trunc = done
#         done = np.abs(info['winner'])
#         reward = info['winner']
#         return obs1, obs2, reward, done, trunc
      
#     @staticmethod
#     def augment(observation: np.ndarray, config) -> np.ndarray:
#         observation_augmented = np.empty(config.env.obs_augmentation_dim)
#         observation_augmented[: observation.shape[0]] = observation

#         player_1 = observation[0:2]
#         player_2 = observation[6:8]
#         puck = observation[12:14]
#         goal_1 = np.array([W / 2 - 250 / SCALE, H / 2])
#         goal_2 = np.array([W / 2 + 250 / SCALE, H / 2])

#         # Augment by adding distances
#         observation_augmented[18] = euclidean_distance(player_1, player_2)
#         observation_augmented[19] = euclidean_distance(player_1, puck)
#         observation_augmented[20] = euclidean_distance(player_2, puck)
#         observation_augmented[21] = euclidean_distance(player_1, goal_1)
#         observation_augmented[22] = euclidean_distance(player_1, goal_2)
#         observation_augmented[23] = euclidean_distance(player_2, goal_1)
#         observation_augmented[24] = euclidean_distance(player_2, goal_2)
#         observation_augmented[25] = euclidean_distance(puck, goal_1)
#         observation_augmented[26] = euclidean_distance(puck, goal_2)

#         return observation_augmented


# # Puck trajectory prediction
# def set_state(self, state):
#     """ function to revert the state of the environment to a previous state (observation)"""
#     self.player1.position = (state[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
#     self.player1.angle = state[2]
#     self.player1.linearVelocity = [state[3], state[4]]
#     self.player1.angularVelocity = state[5]
#     self.player2.position = (state[[6, 7]] + [CENTER_X, CENTER_Y]).tolist()
#     self.player2.angle = state[8]
#     self.player2.linearVelocity = [state[9], state[10]]
#     self.player2.angularVelocity = state[11]
#     self.puck.position = (state[[12, 13]] + [CENTER_X, CENTER_Y]).tolist()
#     self.puck.linearVelocity = [state[14], state[15]]
#     self.player1_has_puck = state[16]
#     self.player2_has_puck = state[17]


# @staticmethod
# def forecast_puck_trajectory(input_obs, n_steps, frame_skip):
#     env = EnvWrapper()
#     env.reset()
#     env.env.set_state(input_obs)
#     env.env.player1.linearVelocity=(0, 0)
#     env.env.player2.linearVelocity=(0, 0)
#     possession1 = input_obs[16] > 0
#     possession2 = input_obs[17] > 0
#     if possession1:
#         action = [0, 0, 0, 1, 0, 0, 0, 0]
#     elif possession2:
#         action = [0, 0, 0, 0, 0, 0, 0, 1]
#     elif env.env.puck.linearVelocity[0] == env.env.puck.linearVelocity[1] == 0:
#         return np.array([(input_obs[12], input_obs[13]) * n_steps]).flatten()
#     else:
#         action = [0, 0, 0, 0, 0, 0, 0, 0]
#     trajectory = []
#     for i in range(n_steps * frame_skip):
#         obs1, obs2, reward, done, trunc = env.step(action)
#     if i % frame_skip == 0:
#         trajectory.append(obs1[12])
#         trajectory.append(obs1[13])

#     return np.array(trajectory).flatten()


# Helper functions for the hockey environment
def get_agent_puck_positions(obs):
    """Extracts agent and puck positions from the observation array.
    Args:
        obs (np.ndarray): Observation array from the environment.
    Returns:
        agent_x (float): X-coordinate of the agent (Player 1).
        agent_y (float): Y-coordinate of the agent (Player 1).
        puck_x (float): X-coordinate of the puck.
        puck_y (float): Y-coordinate of the puck.
    """
    agent_x = obs[0] + CENTER_X
    agent_y = obs[1] + CENTER_Y
    # Correct indices based on environment structure:
    puck_x = obs[12] + CENTER_X  # Position X
    puck_y = obs[13] + CENTER_Y  # Position Y
    puck_vx = obs[14]            # Velocity X
    puck_vy = obs[15]            # Velocity Y
    return agent_x, agent_y, puck_x, puck_y, puck_vx, puck_vy


def predict_puck_position(obs, steps=2, damping=0.97):
    """
    Predict puck position using simple physics model with velocity damping
    Returns: (pred_x, pred_y) in absolute coordinates
    """
    _, _, puck_x, puck_y, vx, vy = get_agent_puck_positions(obs)
    
    pred_x, pred_y = puck_x, puck_y
    current_vx, current_vy = vx, vy
    
    for _ in range(steps):
        # Apply damping to velocity
        current_vx *= damping
        current_vy *= damping
        
        # Update position
        pred_x += current_vx
        pred_y += current_vy
        
    return pred_x, pred_y

def speed(v):
    """Calculates the speed of a 2D vector."""
    return np.linalg.norm(v)

def normalize_vector(v):
    """Normalizes a 2D vector."""
    norm = speed(v)
    if norm == 0:
        return v
    return v / norm

# 0  x pos player one
# 1  y pos player one
# 2  angle player one
# 3  x vel player one
# 4  y vel player one
# 5  angular vel player one
# 6  x player two
# 7  y player two
# 8  angle player two
# 9 y vel player two
# 10 y vel player two
# 11 angular vel player two
# 12 x pos puck
# 13 y pos puck
# 14 x vel puck
# 15 y vel puck
# Keep Puck Mode
# 16 time left player has puck
# 17 time left other player has puck

def compute_defensive_reward(obs, current_step):
    """Computes sparse rewards for defensive positioning relative to goal area."""
    step_reward = 0.0
    agent_x, agent_y, puck_x, puck_y, vx, vy = get_agent_puck_positions(obs)
    # goal is polygon so we need to get closest points 
    # Compute the front face x coordinate of player1's goal.
    goal_front_x = CENTER_X - 245/SCALE
    # Define the vertical boundaries of the goal opening.
    goal_top_y = (CENTER_Y + GOAL_SIZE/SCALE)  # cirlcle to rectangle
    goal_bottom_y = (CENTER_Y - GOAL_SIZE/SCALE)  # cirlcle to rectangle 

    # Clamp the puck's y coordinate to the goal's vertical range.
    closest_goal_y = min(max(puck_y, goal_bottom_y), goal_top_y)
    # The closest point on the goal's front face:
    closest_goal_point = np.array([goal_front_x, closest_goal_y])


    # dist to puck and dist to goal
    dist_to_puck = np.sqrt((agent_x - puck_x)**2 + (agent_y - puck_y)**2)
    dist_to_goal = np.sqrt((agent_x - closest_goal_point[0])**2 + (agent_y - closest_goal_point[1])**2)
    dist_puck_to_goal = np.sqrt((puck_x - closest_goal_point[0])**2 + (puck_y - closest_goal_point[1])**2)
    print("Distance Agent to Puck: ", dist_to_puck, "Distance Agent to Goal: ", dist_to_goal, "Distance Puck to Goal: ", dist_puck_to_goal)
    
    # speed puck
    puck_velocity = np.array([vx, vy])
    puck_speed = speed(puck_velocity)
    puck_velocity_norm = normalize_vector(puck_velocity)

    # Calculate vectors for positioning
    puck_to_goal = np.array([closest_goal_point[0] - puck_x, closest_goal_point[1] - puck_y])
    puck_to_goal_norm = normalize_vector(puck_to_goal)

    agent_has_puck = obs[16] > 0

    # Check dot product of puck velocity and goal-to-puck vector
    alignment_puck_goal = np.dot(puck_to_goal, puck_velocity_norm)
    # Compute alignment: if close to -1, agent is directly between puck and goal
    agent_from_puck = np.array([agent_x - puck_x, agent_y - puck_y])
    alignment_agent = np.dot(normalize_vector(agent_from_puck), puck_to_goal_norm)
    
    if alignment_puck_goal > 0.25 and not agent_has_puck :  # Puck flying towards goal and agent doesnt have puck
        # Define a maximum distance beyond which the proximity reward is zero
        max_puck_distance = 500 / SCALE  # adjust this threshold as needed
        # Calculate a normalized proximity factor (closer = higher value)
        puck_proximity = 1 - min(dist_puck_to_goal / max_puck_distance, 1)
        # Reward more when the puck is closer to the goal, weighted by its speed
        puck_proximity *= puck_speed

        # goal proximit reward
        goal_proximity = 1 - min(dist_to_goal / (250/SCALE), 1)
        step_reward += goal_proximity * puck_proximity * 0.15
        #print("Puck flying towards goal, reward for goal proximity: ", step_reward)

        if alignment_agent > 0.95:
            step_reward += 0.3 * puck_speed  # strong reward for perfect blocking, bonus for fast move
            print("PERFEKT:                  ", step_reward)
            # print puck pos, agent pos, and alignment in one listing
            #print("Puck pos: ", puck_x, puck_y, "Agent pos: ", agent_x, agent_y, "Alignment: ", alignment_agent)
        if alignment_agent < 0.75:
            step_reward += 0.1  * puck_speed # partial reward for perfect blocking
            print("PARTIAL:          ", step_reward)
        # Minus reward for not blocking
        else:
            step_reward -= 0.2 * puck_proximity
            print("NOT: ", step_reward)

    return step_reward

def compute_defensive_reward_orig(obs):
    """Computes sparse rewards for defensive positioning relative to goal area."""
    step_reward = 0.0

    agent_x, agent_y, puck_x, puck_y, vx, vy = get_agent_puck_positions(obs)
    pred_x, pred_y = predict_puck_position(obs)
    
    # Calculate key positions
    goal_x = CENTER_X - 210/SCALE #245
    goal_y = CENTER_Y
    current_dist_to_goal = np.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)
    
    # Calculate distances to predicted puck trajectory
    dist_to_current_puck = np.sqrt((agent_x - puck_x)**2 + (agent_y - puck_y)**2)
    dist_to_future_puck = np.sqrt((agent_x - pred_x)**2 + (agent_y - pred_y)**2)
    
    step_reward = 0.0
    
    # Base positioning reward
    if puck_x > CENTER_X:  # Puck in offensive zone
        goal_proximity = 1 - min(current_dist_to_goal / (250/SCALE), 1)
        step_reward += goal_proximity * 0.5 
        
    # Future puck interception bonus
    if pred_x < CENTER_X:  # Puck entering defensive zone
        future_puck_proximity = 1 - min(dist_to_future_puck / (300/SCALE), 1)
        step_reward += future_puck_proximity * 0.3 
        
    # Penalty for ignoring puck trajectory
    if dist_to_future_puck > 150/SCALE and pred_x < CENTER_X:
        step_reward -= 0.2 
        
    # Existing defensive positioning logic
    if (puck_x > CENTER_X and agent_x > CENTER_X - 75/SCALE):
        step_reward -= 0.1

    return step_reward



    # # Might changing centre/puck migt be better than agent
    # # Reward for staying near goal center (both X and Y)
    # if puck_x > CENTER_X:  # In offensive half
    #     # Base positioning reward (stronger near goal center)
    #     goal_proximity = 1 - min(dist_to_goal / (250/SCALE), 1)
    #     step_reward += goal_proximity * 0.5
        
    # # Extra reward for being between puck and goal when puck is in defensive zone
    # if puck_x < CENTER_X:
    #     # Use similar distance scaling as original closeness reward
    #     max_dist = 250/SCALE
    #     puck_proximity_factor = 1 - min(dist_to_puck/max_dist, 1)
    #     step_reward += puck_proximity_factor * 0.2
        
    #     # Penalize being far from puck in defensive zone (mirror original negative reward)
    #     step_reward -= (dist_to_puck/max_dist) * 0.1

    # # Penalize leaving defensive position unnecessarily when enemy is on other side
    # if ( puck_x > CENTER_X and agent_x > CENTER_X - 100/SCALE):  # Too far in middle line
    #     step_reward -= 0.1

    # return step_reward

# clean empty data folders
def clean_empty_dirs(data_path):
    """
    Recursively removes all folders within the specified directory that are empty.
    
    Args:
        data_path (str): Path to the root directory to clean.
    """
    # Walk through the directory structure from bottom to top
    for root, dirs, _ in os.walk(data_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            
            # Check if the directory is empty
            if not os.listdir(dir_path):
                print(f"Removing empty directory: {dir_path}")
                os.rmdir(dir_path)

    # Check if the root itself becomes empty and clean it
    if not os.listdir(data_path):
        print(f"Removing empty root directory: {data_path}")
        os.rmdir(data_path)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def poll_opponent(opponents):
    return random.choice(opponents)


def dist_positions(p1, p2):
    return np.sqrt(np.sum(np.asarray(p1 - p2) ** 2, axis=-1))


def compute_reward_closeness_to_puck(transition):
    observation = np.asarray(transition[2])
    reward_closeness_to_puck = 0
    if (observation[-6] + CENTER_X) < CENTER_X and observation[-4] <= 0:
        dist_to_puck = dist_positions(observation[:2], observation[-6:-4])
        max_dist = 250. / SCALE
        max_reward = -30.  # max (negative) reward through this proxy
        factor = max_reward / (max_dist * 250 / 2)
        reward_closeness_to_puck += dist_to_puck * factor  # Proxy reward for being close to puck in the own half

    return reward_closeness_to_puck


def compute_winning_reward(transition, is_player_one):
    r = 0

    if transition[4]:
        if transition[5]['winner'] == 0:  # tie
            r += 0
        elif transition[5]['winner'] == 1 and is_player_one:  # you won
            r += 10
        elif transition[5]['winner'] == -1 and not is_player_one:
            r += 10
        else:  # opponent won
            r -= 10
    return r


def recompute_rewards(match, username):
    transitions = match['transitions']
    is_player_one = match['player_one'] == username
    new_transitions = []
    for transition in transitions:
        new_transition = list(deepcopy(transition))
        new_transition[3] = compute_winning_reward(transition, is_player_one) + \
            compute_reward_closeness_to_puck(transition)
        new_transition[5]['reward_closeness_to_puck']
        new_transitions.append(tuple(new_transition))

    return new_transitions


class Logger:
    """
    The Logger class is used printing statistics, saving/loading models and plotting.

    Parameters
    ----------
    prefix_path : Path
        The variable is used for specifying the root of the path where the plots and models are saved.
    mode: str
        The variable specifies in which mode we are currently running. (shooting | defense | normal)
    cleanup: bool
        The variable specifies whether the logging folder should be cleaned up.
    quiet: boolean
        This variable is used to specify whether the prints are hidden or not.
    
    suffix: str
        The variable specifies the suffix of the files to be saved.
    """

    def __init__(self, prefix_path, mode, cleanup=False, quiet=False, suffix=".png") -> None: # Add suffix parameter
        self.prefix_path = Path(prefix_path)

        self.agents_prefix_path = self.prefix_path.joinpath('agents')
        self.plots_prefix_path = self.prefix_path.joinpath('plots')
        self.arrays_prefix_path = self.prefix_path.joinpath('arrays')

        self.prefix_path.mkdir(exist_ok=True)
        self.suffix = suffix

        if cleanup:
            self._cleanup()

        self.quiet = quiet

        if not self.quiet:
            print(f"Running in mode: {mode}")

    def info(self, message):
        print(message)

    def save_model(self, model, filename):
        savepath = self.agents_prefix_path.joinpath(filename).with_suffix('.pkl')
        with open(savepath, 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    def print_episode_info(self, game_outcome, episode_counter, step, total_reward, epsilon=None, touched=None,
                           opponent=None):
        if not self.quiet:
            padding = 8 if game_outcome == 0 else 0
            msg_string = '{} {:>4}: Done after {:>3} steps. \tReward: {:<15}'.format(
                " " * padding, episode_counter, step + 1, round(total_reward, 4))

            if touched is not None:
                msg_string = '{}Touched: {:<15}'.format(msg_string, int(touched))

            if epsilon is not None:
                msg_string = '{}Eps: {:<5}'.format(msg_string, round(epsilon, 2))

            if opponent is not None:
                msg_string = '{}\tOpp: {}'.format(msg_string, opponent)

            print(msg_string)

    def print_stats(self, rew_stats, touch_stats, won_stats, lost_stats):
        if not self.quiet:
            print(tabulate([['Mean reward', np.around(np.mean(rew_stats), 3)],
                            ['Mean touch', np.around(np.mean(list(touch_stats.values())), 3)],
                            ['Mean won', np.around(np.mean(list(won_stats.values())), 3)],
                            ['Mean lost', np.around(np.mean(list(lost_stats.values())), 3)]], tablefmt='grid'))

    def load_model(self, filename):
        if filename is None:
            load_path = self.agents_prefix_path.joinpath('agent.pkl')
        else:
            load_path = Path(filename)
        with open(load_path, 'rb') as inp:
            return pickle.load(inp)

    def hist(self, data, title, filename=None, show=True):
        plt.figure()
        plt.hist(data, density=True)
        plt.title(title)

        plt.savefig(self.reward_prefix_path.joinpath(filename).with_suffix(self.suffix))
        if show:
            plt.show()
        plt.close()

    def plot_running_mean(self, data, title, filename=None, show=True, v_milestones=None, window=250):
        data_np = np.asarray(data)
        mean = running_mean(data_np, window)
        self._plot(mean, title, filename, show)

    def plot_evaluation_stats(self, data, eval_freq, filename):
        style = {
            'weak': 'dotted',
            'strong': 'solid'
        }

        xlen = 0
        for opponent in data.keys():
            stats = data[opponent]
            xlen = len(stats['won'])
            x = np.arange(eval_freq, eval_freq * xlen + 1, eval_freq)
            plt.plot(
                x,
                stats['won'],
                label=f'Won vs {opponent} opponent',
                color='blue',
                linestyle=style[opponent]
            )
            plt.plot(
                x,
                stats['lost'],
                label=f'Lost vs {opponent} opponent',
                color='red',
                linestyle=style[opponent]
            )

            self.to_csv(stats['won'], f'{opponent}_won')

        ticks = labels = np.arange(eval_freq, eval_freq * xlen + 1, eval_freq)
        plt.xticks(ticks, labels, rotation=45)
        plt.ylim((0, 1))
        plt.xlim((eval_freq, xlen * eval_freq))
        plt.title('Evaluation statistics')
        plt.xlabel('Number of training episodes')
        plt.ylabel('Percentage of lost/won games in evaluation')

        lgd = plt.legend(bbox_to_anchor=(1.5, 1))
        plt.savefig(
            self.plots_prefix_path.joinpath(filename).with_suffix(self.suffix),
            bbox_extra_artists=(lgd,),
            bbox_inches='tight'
        )
        plt.close()

    def plot(self, data, title, filename=None, show=True):
        self._plot(data, title, filename, show)

    def plot_intermediate_stats(self, data, show=True):
        self._plot((data["won"], data["lost"]), "Evaluation won vs loss", "evaluation-won-loss", show, ylim=(0, 1))

        for key in data.keys() - ["won", "lost"]:
            title = f'Evaluation {key} mean'
            filename = f'evaluation-{key}.pdf'

            self._plot(data[key], title, filename, show)

    def _plot(self, data, title, filename=None, show=True, ylim=None, v_milestones=None):
        plt.figure()
        # Plotting Won vs lost
        if isinstance(data, tuple):
            plt.plot(data[0], label="Won", color="blue")
            plt.plot(data[1], label="Lost", color='red')
            plt.ylim(*ylim)
            plt.legend()
        else:
            plt.plot(data)
        plt.title(title)

        if v_milestones is not None:
            plt.vlines(
                v_milestones,
                linestyles='dashed',
                colors='orange',
                label='Added self as opponent',
                linewidths=0.5,
                ymin=np.min(data),
                ymax=np.max(data)
            )

        plt.savefig(self.plots_prefix_path.joinpath(filename).with_suffix(self.suffix))
        if show:
            plt.show()

        plt.close()

    def to_csv(self, data, filename):
        savepath = self.arrays_prefix_path.joinpath(filename).with_suffix('.csv')
        np.savetxt(savepath, data, delimiter=',')

    def save_array(self, data, filename):
        savepath = self.arrays_prefix_path.joinpath(filename).with_suffix('.pkl')
        with open(savepath, 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    def load_array(self, filename):
        loadpath = self.arrays_prefix_path.joinpath(filename).with_suffix('.pkl')
        with open(loadpath, 'rb') as inp:
            return pickle.load(inp)

    def _cleanup(self):
        shutil.rmtree(self.agents_prefix_path, ignore_errors=True)
        shutil.rmtree(self.plots_prefix_path, ignore_errors=True)
        shutil.rmtree(self.arrays_prefix_path, ignore_errors=True)
        self.agents_prefix_path.mkdir(exist_ok=True)
        self.plots_prefix_path.mkdir(exist_ok=True)
        self.arrays_prefix_path.mkdir(exist_ok=True)
