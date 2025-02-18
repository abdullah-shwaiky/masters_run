import os
import sys
import numpy as np
import torch
import argparse
from collections import defaultdict

# Set up SUMO
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sumo_rl
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Command-line argument parsing for alpha_val
    parser = argparse.ArgumentParser(description='SARSA with SUMO simulation')
    parser.add_argument('--alpha_val', type=float, default=0.7, help='Learning rate for SARSA')
    args = parser.parse_args()

    alpha_val = args.alpha_val  # Learning rate for SARSA passed via command line
    alpha = 0.01  # Fixed alpha for SARSA (do not change)
    gamma = 0.99  # Discount factor
    epsilon = 0.05  # Epsilon for epsilon-greedy exploration
    runs = 25  # Number of training runs

    from my_maps import map_details

    for map_ in map_details:
        env = sumo_rl.env(
            net_file=map_["net_file"],
            route_file=map_["route_file"],
            use_gui=False,
            num_seconds=2000,
            reward_fn="weighted",
            fixed_ts=False,
            alpha = alpha_val,
            out_csv_name=f"{alpha_val}/{map_['save_location']}sarsa/sarsa",
        )
        env.reset()

        # Initialize SARSA agents with alpha_val
        sarsa_agents = {
            ts: TrueOnlineSarsaLambda(
                state_space=env.observation_space(ts),
                action_space=env.action_space(ts),
                alpha=alpha_val,  # Use alpha_val here
                gamma=gamma,
                epsilon=epsilon,
            )
            for ts in env.agents
        }
        total_reward = 0
        for run in range(1, runs + 1):
            print(f"Starting run {run}...")
            env.reset()
            observations = {ts: env.observe(ts) for ts in env.agents}
            terminated_agents = set()

            while len(terminated_agents) < len(env.agents):
                actions = {
                    ts: sarsa_agents[ts].act(obs)
                    for ts, obs in observations.items()
                    if ts not in terminated_agents
                }

                # Step in the environment for each agent
                for ts, action in actions.items():
                    env.step(action)

                # Collect experiences
                next_observations = {}
                rewards = {}
                dones = {}
                for ts in env.agents:
                    if ts in terminated_agents:
                        continue
                    obs, reward, terminated, truncated, _ = env.last(ts)
                    next_observations[ts] = obs
                    rewards[ts] = reward
                    dones[ts] = terminated or truncated
                    if dones[ts]:
                        terminated_agents.add(ts)

                # Update SARSA agents
                total_reward += sum(rewards.values())
                for ts in actions.keys():
                    if not dones[ts]:
                        next_action = sarsa_agents[ts].act(next_observations[ts])
                        sarsa_agents[ts].learn(
                            observations[ts],
                            actions[ts],
                            rewards[ts],
                            next_observations[ts],
                            dones[ts],
                        )
                        actions[ts] = next_action

                observations = next_observations
            print(f"Run {run} completed.")
        env.close()
    print(total_reward)
