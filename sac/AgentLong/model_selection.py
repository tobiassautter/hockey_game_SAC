import sys
sys.path.insert(0, '.')
import os
import argparse
import numpy as np
import pandas as pd
import omegaconf
from AgentLong import hockey_env as h_env
from AgentLong.hockey_agent import HockeyAgent
from AgentLong.utils import get_config

N_GAMES = 100

def evaluate(agent1, agent2, n_games=N_GAMES):
    config1 = agent1.config
    config2 = agent2.config

    envs = [h_env.EnvWrapper(config1) for _ in range(n_games)]
    obs1_list, obs2_list, forecast1_list, forecast2_list, winners = [], [], [], [], []

    for seed, env in enumerate(envs):
        obs1, obs2 = env.reset(seed=seed)
        if config1.env.obs_augmentation:
            obs1 = h_env.EnvWrapper.augment(obs1, config1)
        if config2.env.obs_augmentation:
            obs2 = h_env.EnvWrapper.augment(obs2, config2)
        if config1.env.use_forecast:
            forecast1_list.append(h_env.EnvWrapper.forecast(obs1, config1.env.forecast_step, config1.rl.frame_skip))
        if config2.env.use_forecast:
            forecast2_list.append(h_env.EnvWrapper.forecast(obs2, config2.env.forecast_step, config2.rl.frame_skip))
        obs1_list.append(obs1)
        obs2_list.append(obs2)

    while envs:
        obs1_array = np.array(obs1_list).reshape(-1, config1.env.obs_dim)
        obs2_array = np.array(obs2_list).reshape(-1, config2.env.obs_dim)
        forecast1_array = np.array(forecast1_list).reshape(obs1_array.shape[0], -1) if forecast1_list else None
        forecast2_array = np.array(forecast2_list).reshape(obs2_array.shape[0], -1) if forecast2_list else None

        action1_array, _ = agent1.batch_model_search(obs1_array, forecast1_array)
        action2_array, _ = agent2.batch_model_search(obs2_array, forecast2_array)

        for i, env in enumerate(envs):
            obs1, obs2, reward, _, trunc = env.step(np.hstack([action1_array[i], action2_array[i]]))
            if trunc:
                winners.append(reward)
            else:
                if config1.env.obs_augmentation:
                    obs1 = h_env.EnvWrapper.augment(obs1, config1)
                if config2.env.obs_augmentation:
                    obs2 = h_env.EnvWrapper.augment(obs2, config2)
                if config1.env.use_forecast:
                    forecast1_list[i] = h_env.EnvWrapper.forecast(obs1, config1.env.forecast_step, config1.rl.frame_skip)
                if config2.env.use_forecast:
                    forecast2_list[i] = h_env.EnvWrapper.forecast(obs2, config2.env.forecast_step, config2.rl.frame_skip)
                obs1_list[i] = obs1
                obs2_list[i] = obs2

        obs1_list = [obs1 for i, obs1 in enumerate(obs1_list) if not envs[i].env.done]
        obs2_list = [obs2 for i, obs2 in enumerate(obs2_list) if not envs[i].env.done]
        forecast1_list = [forecast1 for i, forecast1 in enumerate(forecast1_list) if not envs[i].env.done]
        forecast2_list = [forecast2 for i, forecast2 in enumerate(forecast2_list) if not envs[i].env.done]
        envs = [env for env in envs if not env.env.done]

    winners = np.array(winners)
    wins = (winners == 1).sum()
    losses = (winners == -1).sum()
    return wins / (wins + losses) if (wins + losses) > 0 else 0.5

def main(input_dir, mixed_configs):
    model_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pth")]
    model_names = [os.path.basename(m) for m in model_paths]
    n = len(model_paths)
    ranking_path = os.path.join(input_dir, "ranking.csv")

    # Load existing ranking if available
    if os.path.exists(ranking_path):
        df = pd.read_csv(ranking_path, index_col=0)
        existing_models = df.index.tolist()
        win_matrix = df.to_numpy()
    else:
        win_matrix = np.full((n, n), np.nan)  # Use NaN to track missing evaluations

    # Extend matrix if new models are added
    if len(model_names) > len(existing_models):
        new_size = len(model_names)
        extended_matrix = np.full((new_size, new_size), np.nan)
        extended_matrix[: len(existing_models), : len(existing_models)] = win_matrix
        win_matrix = extended_matrix
        existing_models = model_names

    # Load config(s)
    if not mixed_configs:
        config_file = next(f for f in os.listdir(input_dir) if f.endswith(".yaml"))
        config = get_config()
        config = omegaconf.OmegaConf.merge(config, omegaconf.OmegaConf.load(os.path.join(input_dir, config_file)))
    else:
        config = None  # Per-model configs will be loaded dynamically

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(win_matrix[i, j]):  # Skip if already evaluated
                continue

            # Load agents with the correct config
            if mixed_configs:
                config1 = get_config()
                config1 = omegaconf.OmegaConf.merge(config1, omegaconf.OmegaConf.load(model_paths[i].replace(".pth", ".yaml")))
                agent1 = HockeyAgent(model_paths[i], config1)

                config2 = get_config()
                config2 = omegaconf.OmegaConf.merge(config2, omegaconf.OmegaConf.load(model_paths[j].replace(".pth", ".yaml")))
                agent2 = HockeyAgent(model_paths[j], config2)
            else:
                agent1 = HockeyAgent(model_paths[i], config)
                agent2 = HockeyAgent(model_paths[j], config)

            # Evaluate only if missing
            win_matrix[i, j] = evaluate(agent1, agent2)
            win_matrix[j, i] = 1 - win_matrix[i, j]

            # Compute win rates for sorting
            win_rates = np.nanmean(win_matrix, axis=1)
            sorted_indices = np.argsort(-win_rates)  # Sort descending

            # Sort rows and columns
            sorted_names = [model_names[k] for k in sorted_indices]
            win_matrix = win_matrix[sorted_indices, :][:, sorted_indices]

            # Save after each evaluation to ensure progress is recorded
            df = pd.DataFrame(win_matrix, index=sorted_names, columns=sorted_names)
            df.to_csv(ranking_path)
            print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory containing models and configs")
    parser.add_argument("--mixed_configs", action="store_true", help="Set this flag if models have different configs, for example when evaluate a candidate pool with different training parameters. Here each model must has an own config.")
    args = parser.parse_args()

    main(args.input_dir, args.mixed_configs)
