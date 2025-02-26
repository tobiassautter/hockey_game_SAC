# hockey-env

This repository is an upgraded SAC version from: https://github.com/anticdimi/laser-hockey 
It includes meta-tuning [Y. Wang ‘20], updated models, observation normalization, modified rewards for stronger defense, RND networks [Burda ‘18], usage of adamW instead of adam and a lot more..
It achieved place 28th out of 148 in the reinforcement learning project.

Bot strong:
![Agent vs bot strong](assets/hockey_SAC_strong_bot.gif)

Bot weak:
![Agent vs weak strong](assets/hockey_SAC_weak_bot.gif)

## Install

Install requirements.txt, it has all the latest libs used.

Hockey-ENV from: 
``python3 -m pip install git+https://github.com/martius-lab/hockey-env.git``

or add the following line to your Pipfile

``hockey = {editable = true, git = "https://git@github.com/martius-lab/hockey-env.git"}``

## Training
One training prompt for a base model was:
python.exe .\sac\train_agent.py --mode normal --learning_rate 0.0005 --lr_milestones=10000 --alpha_milestones=10000 --gamma 0.98 --alpha 0.5 --selfplay True --show_percent 1 --beta 0.15 --beta_end 0.15 --adamw True --meta_tuning  


## Evaluation
Evaluation args could be:
python .\sac\evaluate_agent.py --mode normal --filename .\sac\latest_agent\o6-agent.pkl --eval_episodes 1000 --show_percent 0 --show   

## HockeyEnv (University Tuebingen)
``hockey.hockey_env.HockeyEnv``

A two-player (one per team) hockey environment.
For our Reinforcment Learning Lecture @ Uni-Tuebingen.
See Hockey-Env.ipynb notebook on how to run the environment.

The environment can be generated directly as an object or via the gym registry:

``env = gym.envs.make("Hockey-v0")``

There is also a version against the basic opponent (with options)

``env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)``

