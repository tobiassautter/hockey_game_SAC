from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np

import sys
sys.path.insert(0, 'sac')
from sac.sac_agent_rnd import SACAgent

from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

# TODO Implement the SACWRAPPER agent
class SACWrapper(Agent):
    """A hockey agent that wraps the trained SAC agent."""

    def __init__(self, agent: SACAgent) -> None: 
        super().__init__()
        self.agent = agent

    def get_step(self, observation: list[float]) -> list[float]:
        return self.agent.act(observation).tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "ElSacko"],
        default="weak",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "ElSacko":
        trained_agent = SACAgent.load_model("sac/latest_agent/m-agent.pkl") #f2 #e #d-agent.pkl") #b-6000.pkl")
        agent = SACWrapper(trained_agent)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
# export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
# export COMPRL_SERVER_PORT=65335
# export COMPRL_ACCESS_TOKEN=916cdb8c-bd47-4e2b-b86b-7d0c9e59b1c8
# python.exe run_client.py --server-url comprl.cs.uni-tuebingen.de --server-port 65335 --token 916cdb8c-bd47-4e2b-b86b-7d0c9e59b1c8 --args --agent ElSacko 