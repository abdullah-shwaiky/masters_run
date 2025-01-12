import argparse
import os
import sys
import numpy as np
import random
from collections import deque

# Set up SUMO
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


import sumo_rl
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

if __name__ == "__main__":
    # Hyperparameters for SARSA
    alpha = 0.001  # Learning rate for SARSA
    gamma = 0.99  # Discount factor
    epsilon = 0.05  # Epsilon for epsilon-greedy exploration
    batch_size = 64  # Batch size for experience replay
    runs = 2  # Number of training runs
    
    from my_maps import map_details
    for map_ in map_details:
        env = sumo_rl.env(
            net_file=map_['net_file'],
            route_file=map_['route_file'],
            use_gui=False,
            num_seconds=2000,
            reward_fn="weighted",
            fixed_ts=True,
            out_csv_name=f"{map_['save_location']}fixed/fixed"
        )
        env.reset()

        # Initialize SARSA agents using linear_rl's SARSA class
        sarsa_agents = {
            ts: TrueOnlineSarsaLambda(
                state_space=env.observation_space(ts),  # Assuming state is a 1D array
                action_space=env.action_space(ts),
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
            )
            for ts in env.agents
        }

        for run in range(1, runs + 1):
            env.reset()
            counter = 0

            # Start the simulation loop for each agent
            for agent in env.agent_iter():
                s, r, terminated, truncated, info = env.last()
                done = terminated or truncated

                next_state = env.observe(agent)

                # The agent selects the action using SARSA (epsilon-greedy policy)
                action = sarsa_agents[agent].act(s) if not done else None


                # Move the simulation one step forward
                env.step(action)
                counter += 1
                print(f"Step {counter} for agent {agent}")

            # Save the results for each run
            # env.unwrapped.env.save_csv(f"{map_['save_location']}sarsa/sarsa", run)
            env.close()
