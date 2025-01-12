import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Set up SUMO
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

import sumo_rl
from sumo_rl.exploration import EpsilonGreedy


# DDQN Model (Neural Network to approximate Q-values)
class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=0.0001, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=50)  # Experience Replay buffer
        self.model = DDQN(state_size, action_size).float()  # Q-network
        self.target_model = DDQN(state_size, action_size).float()  # Target Q-network
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Get Q values for current states (from the main Q-network)
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get action indices for the next states (from the main Q-network)
        next_actions = torch.argmax(self.model(next_states), dim=1)

        # Get Q values for next states using the target Q-network
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        # Compute the target Q-values using the Double DQN update rule
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


if __name__ == "__main__":
    alpha = 0.01  # Learning rate for DDQN
    gamma = 0.99
    epsilon = 0.05
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 64
    runs = 25
    from my_maps import map_details
    for map_ in map_details:
        env = sumo_rl.env(
            net_file=map_['net_file'],
            route_file=map_['route_file'],
            use_gui=False,
            num_seconds=2000,
            reward_fn = "weighted",
            fixed_ts = False 
        )

        for run in range(1, runs + 1):
            env.reset()
            initial_states = {ts: env.observe(ts) for ts in env.agents}
            
            ddqn_agents = {
                ts: DDQNAgent(
                    state_size=env.observation_space(ts).shape[0],  # Assuming state is a 1D array
                    action_size=env.action_space(ts).n,
                    gamma=gamma,
                    alpha=alpha,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    batch_size=batch_size
                )
                for ts in env.agents
            }

            counter = 0
            for agent in env.agent_iter():
                s, r, terminated, truncated, info = env.last()
                done = terminated or truncated

                next_state = env.observe(agent) 
                
                action = ddqn_agents[agent].act(s) if not done else None
                if not done:
                    ddqn_agents[agent].learn(s, action, r, next_state, done)

                env.step(action)
                counter += 1
                print(counter)

            env.unwrapped.env.save_csv(f"{map_['save_location']}ddqn/ddqn", run)
            torch.save(ddqn_agents[agent].model.state_dict(), f"models/ddqn/{map_['net_file'].split('/')[-1][:map_['net_file'].split('/')[-1].index('.')]}/model_run_{run}.pth")
            env.close()
