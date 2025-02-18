import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import shutil
from tabulate import tabulate
import random
from copy import deepcopy
from hockey.hockey_env import CENTER_X, CENTER_Y, SCALE, W
import os

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
    # Positions are relative to center (CENTER_X, CENTER_Y)
    agent_x = obs[0] + CENTER_X
    agent_y = obs[1] + CENTER_Y
    puck_x = obs[14] + CENTER_X
    puck_y = obs[15] + CENTER_Y
    return agent_x, agent_y, puck_x, puck_y

# def compute_defensive_reward(agent_x, agent_y, puck_x, puck_y, env_width=W):
#     """Computes sparse rewards for defensive positioning and puck proximity.
#     Args:
#         agent_x (float): Agent's X-coordinate.
#         puck_x (float): Puck's X-coordinate.
#         puck_y (float): Puck's Y-coordinate.
#         env_width (float): Total width of the environment.
#     Returns:
#         step_reward (float): Sparse reward component.
#     """
#     step_reward = 0.0
    
#     # Reward defensive positioning (agent in left half)
#     if agent_x < CENTER_X:
#         step_reward += 0.5  # Base reward for staying in defensive zone
    
#     # Extra reward if near the goal area (leftmost 20% of the rink)
#     if agent_x < 0.2 * env_width:
#         step_reward += 0.3 * 10
    
#     # Penalize over-aggressiveness (crossing to opponent's half)
#     if agent_x > CENTER_X + 0.2 * env_width:
#         step_reward -= 0.2 * 10
    
#     # Reward puck interception (if puck is in defensive zone)
#     if puck_x < CENTER_X:
#         puck_dist = np.sqrt((agent_x - puck_x)**2 + (agent_y - puck_y)**2)
#         if puck_dist < 0.1 * env_width:  # Close to puck
#             step_reward += 0.4
    
#     return step_reward
def compute_defensive_reward(agent_x, agent_y, puck_x, puck_y):
    """Computes sparse rewards for defensive positioning relative to goal area."""
    step_reward = 0.0
    
    # Goal is at left side: (CENTER_X - 245/SCALE, CENTER_Y)
    goal_x = CENTER_X - 245/SCALE
    goal_y = CENTER_Y
    
    # Calculate distances
    dist_to_goal = np.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)
    dist_to_puck = np.sqrt((agent_x - puck_x)**2 + (agent_y - puck_y)**2)
    
    # # Might changing centre/puck migt be better than agent
    # # Reward for staying near goal center (both X and Y)
    # if agent_x < CENTER_X - 50/SCALE:  # In defensive half
    #     # Base positioning reward (stronger near goal center)
    #     goal_proximity = 1 - min(dist_to_goal / (250/SCALE), 1)
    #     step_reward += goal_proximity * 0.35
        
    #     # Extra reward for being between puck and goal when puck is in defensive zone
    #     if puck_x < CENTER_X:
    #         # Use similar distance scaling as original closeness reward
    #         max_dist = 250/SCALE
    #         puck_proximity_factor = 1 - min(dist_to_puck/max_dist, 1)
    #         step_reward += puck_proximity_factor * 0.2
            
    #         # Penalize being far from puck in defensive zone (mirror original negative reward)
    #         step_reward -= (dist_to_puck/max_dist) * 0.1

    # # Penalize leaving defensive position unnecessarily
    # if agent_x > CENTER_X - 75/SCALE:  # Too far in middle line
    #     step_reward -= 0.1
    
    # Might changing centre/puck migt be better than agent
    # Reward for staying near goal center (both X and Y)
    if puck_x > CENTER_X:  # In offensive half
        # Base positioning reward (stronger near goal center)
        goal_proximity = 1 - min(dist_to_goal / (250/SCALE), 1)
        step_reward += goal_proximity * 0.35
        
        # Extra reward for being between puck and goal when puck is in defensive zone
    # if puck_x < CENTER_X:
    #     # Use similar distance scaling as original closeness reward
    #     max_dist = 250/SCALE
    #     puck_proximity_factor = 1 - min(dist_to_puck/max_dist, 1)
    #     step_reward += puck_proximity_factor * 0.2
        
    #     # Penalize being far from puck in defensive zone (mirror original negative reward)
    #     step_reward -= (dist_to_puck/max_dist) * 0.1

    # Penalize leaving defensive position unnecessarily when enemy is on other side
    if ( puck_x > CENTER_X and agent_x > CENTER_X - 75/SCALE):  # Too far in middle line
        step_reward -= 0.1

    return step_reward

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

    def plot_running_mean(self, data, title, filename=None, show=True, v_milestones=None):
        data_np = np.asarray(data)
        mean = running_mean(data_np, 1000)
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
