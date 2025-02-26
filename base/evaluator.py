import time
import numpy as np

# USING BASE SCRIPTS FROM 1. PLACE 2021 COMPETITION
# https://github.com/anticdimi/laser-hockey 

# @modfied 
def evaluate(agent, env, opponent, eval_episodes, show_percentage=10,
            quiet=False, action_mapping=None, evaluate_on_opposite_side=False,
            long_agent=False,
            ):
    old_verbose = env.verbose
    env.verbose = not quiet

    rew_stats = []
    touch_stats = {}
    won_stats = {}
    lost_stats = {}

    for episode_counter in range(eval_episodes):
        total_reward = 0
        ob = env.reset(seed=episode_counter)
        obs_agent2 = env.obs_agent_two()

        if (env.puck.position[0] < 5 and agent._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and agent._config['mode'] == 'shooting'
        ):
            continue

        touch_stats[episode_counter] = 0
        won_stats[episode_counter] = 0
        lost_stats[episode_counter] = 0
        # Evaluate the agent for max_timesteps
        for step in range(env.max_timesteps):
            if action_mapping is not None:
                # DQN act
                a1 = agent.act(ob, eps=0)
                a1 = action_mapping[a1]
            else:
                # SAC act
                a1 = agent.act(ob)
            #print("Long agent: ", long_agent)
            if agent._config['mode'] in ['defense', 'normal'] and not long_agent:
                a2 = opponent.act(obs_agent2)
                if not isinstance(a2, np.ndarray):
                    a2 = action_mapping[a2]
            elif agent._config['mode'] == 'shooting':
                a2 = [0, 0, 0, 0]
            elif long_agent:
                a2 = opponent.get_step(obs_agent2)
                # only first dimension is needed
                a2 = a2[0]

            else:
                raise NotImplementedError(f'Training for {agent._config["mode"]} not implemented.')

            ob_new, reward, done, f, _info = env.step(np.hstack([a1, a2]))
            ob = ob_new
            obs_agent2 = env.obs_agent_two()

            if evaluate_on_opposite_side:
                # Not really a way to implement this, given the structure of the env...
                touch_stats[episode_counter] = 0
                total_reward -= reward

            else:
                if _info['reward_touch_puck'] > 0:
                    touch_stats[episode_counter] = 1

                total_reward += reward

            if agent._config['show']:
                time.sleep(0.001)
                env.render()
            

            # if evaluation show be rendered and steps are divisible by show_percentage
            if  (show_percentage != 0):
                # try to modulo to get percetage, else echo "perecentage too low and set to 0"
                try:
                    show = episode_counter % int(eval_episodes / show_percentage) == 0
                except ZeroDivisionError:
                    show = False
                    #print("Show percentage too low and set to 0")
                if show:
                    #time.sleep(0.002)
                    env.render()
        
            if done:
                if evaluate_on_opposite_side:
                    won_stats[episode_counter] = 1 if env.winner == -1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == 1 else 0
                else:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                break

        rew_stats.append(total_reward)
        if not quiet:
            agent.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon=0,
                                            touched=touch_stats[episode_counter])

    if not quiet:
        # Print evaluation stats
        agent.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

    # Toggle the verbose flag onto the old value
    env.verbose = old_verbose

    return (
        np.mean(rew_stats),
        np.mean(list(touch_stats.values())),
        np.mean(list(won_stats.values())),
        np.mean(list(lost_stats.values()))
    )
