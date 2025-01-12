import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

import pandas as pd

# Set up SUMO
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

import sumo_rl
from sumo_rl.exploration import EpsilonGreedy


# PPO Model (Policy + Value Network)
class PPO(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value


class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=0.0003, epsilon=0.2, batch_size=64, update_epochs=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        self.memory = deque(maxlen=50)  # Experience Replay buffer
        self.model = PPO(state_size, action_size).float()  # Policy and value network
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy, _ = self.model(state)
        action = torch.multinomial(policy, 1).item()  # Sample action from policy
        return action

    def store(self, state, action, reward, next_state, done, log_prob, value):
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def learn(self):
        # Update the policy using PPO's objective
        states, actions, rewards, next_states, dones, log_probs, values = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        _, next_values = self.model(next_states)
        _, values = self.model(states)

        # Compute the target value (TD target)
        target_values = rewards + (self.gamma * next_values.squeeze(1) * (1 - dones.float()))

        # Compute advantages (TD error)
        advantages = target_values - values.squeeze(1)

        # Compute the loss for the policy (surrogate objective with clipping)
        policy, _ = self.model(states)
        new_log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)))
        ratios = torch.exp(new_log_probs - torch.tensor(log_probs))  # ratio of new to old policy

        # Clipped surrogate objective
        clip_adv = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(ratios * advantages, clip_adv).mean()

        # Compute the value loss (MSE loss between predicted value and target value)
        value_loss = nn.MSELoss()(values.squeeze(1), target_values)

        # Total loss is the sum of policy loss and value loss
        loss = policy_loss + 0.5 * value_loss  # 0.5 coefficient for value loss

        # Update the model using backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after each update
        self.memory.clear()


if __name__ == "__main__":
    alpha = 0.01  # Learning rate for PPO
    gamma = 0.99
    epsilon = 0.05
    batch_size = 64
    update_epochs = 10
    runs = 25
    from my_maps import map_details
    for map_ in map_details:
        env = sumo_rl.env(
            net_file=map_['net_file'],
            route_file=map_['route_file'],
            use_gui=False,
            num_seconds=2000,
            reward_fn="weighted",
            fixed_ts=False
        )

        for run in range(1, runs + 1):
            env.reset()
            initial_states = {ts: env.observe(ts) for ts in env.agents}
            
            ppo_agents = {
                ts: PPOAgent(
                    state_size=env.observation_space(ts).shape[0],  # Assuming state is a 1D array
                    action_size=env.action_space(ts).n,
                    gamma=gamma,
                    alpha=alpha,
                    epsilon=epsilon,
                    batch_size=batch_size,
                    update_epochs=update_epochs
                )
                for ts in env.agents
            }

            counter = 0
            for agent in env.agent_iter():
                s, r, terminated, truncated, info = env.last()
                done = terminated or truncated

                next_state = env.observe(agent) 
                
                action = ppo_agents[agent].act(s) if not done else None
                if not done:
                    # Store experience for learning
                    log_prob = torch.log(ppo_agents[agent].model(torch.tensor(s, dtype=torch.float32).unsqueeze(0))[0][0][action])
                    value = ppo_agents[agent].model(torch.tensor(s, dtype=torch.float32).unsqueeze(0))[1]
                    ppo_agents[agent].store(s, action, r, next_state, done, log_prob, value)

                env.step(action)
                counter += 1
                print(counter)

                # Perform learning after each episode
                if done:
                    ppo_agents[agent].learn()

            env.unwrapped.env.save_csv(f"{map_['save_location']}ppo/ppo", run)
            torch.save(ppo_agents[agent].model.state_dict(), f"models/ppo/{map_['net_file'].split('/')[-1][:map_['net_file'].split('/')[-1].index('.')]}/model_run_{run}.pth")
            env.close()
