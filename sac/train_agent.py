import os
import torch
import sys
import sys
sys.path.insert(1, '..')
from hockey import hockey_env as h_env
from sac_agent_rnd import SACAgent
from argparse import ArgumentParser

from trainer import SACTrainer
import time
import random


from utils.utils import *
from base.experience_replay import ExperienceReplay

# USING BASE SCRIPTS FROM 1. PLACE 2021 COMPETITION
# https://github.com/anticdimi/laser-hockey 

parser = ArgumentParser()
parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true')
parser.add_argument('--cuda', help='Set if want to train on graphic card', action='store_true')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='defense')
parser.add_argument('--preload_path', help='Path to the pretrained model', default=None)
parser.add_argument('--transitions_path', help='Path to the root of folder containing transitions', default=None)

# Training params
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=5000) #5000
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=250)
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=50)
parser.add_argument('--evaluate_every',
                    help='# of episodes between evaluating agent during the training', type=int, default=500) #1000
parser.add_argument('--add_self_every',
                    help='# of gradient updates between adding agent (self) to opponent list', type=int, default=30000)#1001)#100000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-3) #1e-3)
parser.add_argument('--alpha_lr', help='Learning rate', type=float, default=1e-4) #1e-4) #For meta : 3e-4
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--alpha_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--update_target_every', help='# of steps between updating target net', type=int, default=1)
parser.add_argument('--gamma', help='Discount', type=float, default=0.95) #0.95)
parser.add_argument('--batch_size', help='batch_size', type=int, default=128) #128
parser.add_argument('--grad_steps', help='grad_steps', type=int, default=32)
parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter alpha determines the relative importance of the entropy term against the reward')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False,
                    help='Automatically adjust alpha')
parser.add_argument('--selfplay', type=bool, default=False, help='Should agent train selfplaf')
parser.add_argument('--soft_tau', help='tau', type=float, default=0.005)
parser.add_argument('--per', help='Utilize Prioritized Experience Replay', action='store_true')
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)
parser.add_argument('--per_beta', help='Beta for PER', type=float, default=0.4)

#extra sauce: ---
# env_render_eval = True
parser.add_argument('--e_r', help='Set if want to see evaluation process rendered', action='store_true')
parser.add_argument('--show_percent', help='Percentage of episodes to show', type=int, default=10)

# RND params
parser.add_argument('--beta', type=float, default=1.0, help='Intrinsic reward scaling factor')
parser.add_argument('--rnd_lr', type=float, default=5e-3, help='Learning rate for RND predictor')

# Train sparse reward
parser.add_argument('--sparse', type=bool, default=False, help='Train with sparse reward')

# train against pretrained agents only
parser.add_argument('--pretrained', type=bool, default=False, help='Train against pretrained agents only')

# Use adam or adamW and adamW params
parser.add_argument('--adamw', type=bool, default=True, help='Use AdamW optimizer')
parser.add_argument('--adamw_eps', type=float, default=1e-6, help='AdamW epsilon')
parser.add_argument('--adamw_weight_decay', type=float, default=1e-6, help='AdamW weight decay')

# Add meta tuning instead of old entropy tuning with meta_batch_size
parser.add_argument('--meta_tuning', action='store_true', help='Use Meta-SAC entropy tuning')
parser.add_argument('--meta_batch_size', type=int, default=32, help='Batch size for meta tuning')
# add meta scale 20.0
parser.add_argument('--meta_scale', type=float, default=50.0, help='Scale for meta tuning')

# add buffer size
parser.add_argument('--buffer_size', type=int, default=10, help='Buffer size for experience replay')
opts = parser.parse_args()

if __name__ == '__main__':
    if opts.dry_run:
        opts.max_episodes = 10

    if opts.mode == 'normal':
        mode = h_env.Mode.NORMAL
    elif opts.mode == 'shooting':
        mode = h_env.Mode.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.Mode.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')

    opts.device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')


    # Define the root data directory
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    # Generate a unique directory name
    dirname = time.strftime(f'%y%m%d_%H%M%S_{random.randint(0, 123456):06}', time.gmtime(time.time()))
    dirname = f'{opts.mode}_{dirname}'  # Combine mode and timestamp

    # Final path for the training data
    training_data_path = os.path.join(data_dir, dirname)

    # Ensure the data directory exists
    os.makedirs(training_data_path, exist_ok=True)

    # Pass the training data path to the logger
    logger = Logger(
        prefix_path=training_data_path,
        mode=opts.mode,
        cleanup=True,
        quiet=opts.q
    )

    env = h_env.HockeyEnv(mode=mode, verbose=(not opts.q))
    opponents = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]

    # Add absolute paths for pretrained agents
    pretrained_agents = [
        #"sac/all_agents/k-agent.pkl",
        #"sac/all_agents/i4-agent.pkl", 
        "sac/all_agents/m-agent.pkl", 
        "sac/all_agents/n-agent.pkl", 
        "sac/all_agents/n2-agent.pkl"
        ]
    
    # Set up the pretrained agents
    if opts.pretrained:
        for p in pretrained_agents:
            a = SACAgent.load_model(p)
            a.eval()
            opponents.append(a)


    if opts.selfplay:
        for p in pretrained_agents:
            a = SACAgent.load_model(p)
            a.eval()
            opponents.append(a)

    if opts.preload_path is None:
        agent = SACAgent(
            logger=logger,
            obs_dim=env.observation_space.shape,
            action_space=env.action_space,
            userconfig=vars(opts)
        )
    else:
        agent = SACAgent.load_model(opts.preload_path)
        if opts.per:
            agent.buffer = ExperienceReplay.clone_buffer_per(agent.buffer, opts.buffer_size, opts.per_alpha, opts.per_beta)
        else:
            agent.buffer = ExperienceReplay.clone_buffer(agent.buffer, opts.buffer_size)#300000)

        #agent.buffer.preload_transitions(opts.transitions_path)
        agent.train()


    # Log File for better parater parsing and overview
    log_file_path = os.path.join(data_dir, dirname, f"{dirname}_debug.log")

    # Open the log file for writing
    with open(log_file_path, "w") as log_file:
        def log_print(*args, **kwargs):
            """Prints to console and writes to a file."""
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=log_file)  # Write to file

        log_print("Data will be saved to:", log_file_path)
        log_print("Debugging info:")

        # Debugging: print agent, opponents, env
        log_print("Agent:           ", agent)
        log_print("Opponents:       ", opponents)
        log_print("env:             ", env)

        # Print agent shape, etc
        log_print("agent shape:     ", agent.obs_dim)
        log_print("ag_action_space: ", agent.action_space)

        # Print env shape, etc
        log_print("env shape:       ", env.observation_space.shape)
        log_print("obs_dim:         ", env.observation_space.shape)
        log_print("action_space:    ", env.action_space)

        # Print opts values
        for key, value in vars(opts).items():
            log_print(f"{key}: {value}")

    print(f"Debugging info saved to: {log_file_path}")


    print("Training agent...")
    trainer = SACTrainer(logger, vars(opts))
    trainer.train(agent, opponents, env, opts.evaluate)

    # clean data folder of empty folders
    # clean_empty_dirs(data_dir)\


# Args options:

# python.exe .\sac\train_agent.py --lr_milestones 500 --evaluate_every 250  --mode normal --max_episodes 10000 --per --automatic_entropy_tuning True --alpha_milestones 100 --learning_rate 1e-4 --gamma 0.98 --update_target_every 5 --max_steps 1000 --batch_size 256 --soft_tau 0.0025 --env_render_eval
# python.exe .\sac\train_agent.py --lr_milestones 200 --evaluate_every 10  --mode normal --max_episodes 200 --per --automatic_entropy_tuning True --alpha_milestones 200 --learning_rate 1e-2 --env_render_eval

# latest run:
# +-------------+--------+
# | Mean reward | -7.385 |
# +-------------+--------+
# | Mean touch  |  0.953 |
# +-------------+--------+
# | Mean won    |  0.58  |
# +-------------+--------+
# | Mean lost   |  0.168 |
# +-------------+--------+
# python.exe .\sac\train_agent.py --lr_milestones 500 --evaluate_every 50  --mode normal --max_episodes 10000 --per --automatic_entropy_tuning True --alpha_milestones 100 --learning_rate 1e-4 --gamma 0.98 --update_target_every 2 --max_steps 5000 --batch_size 256 --soft_tau 0.0025 --env_render_eval


# +-------------+---------+
# | Mean reward | -22.816 |
# +-------------+---------+
# | Mean touch  |   0.882 |
# +-------------+---------+
# | Mean won    |   0.344 |
# +-------------+---------+
# | Mean lost   |   0.327 |
# +-------------+---------+
# 