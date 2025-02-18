import os
import sys
import torch
import torch.nn as nn
import numpy as np
import csv
from collections import deque
from torch.distributions import Categorical
import argparse  # Import argparse for handling command-line arguments

# Set up SUMO
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sumo_rl
import traci


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# TRPO Update Functions
def conjugate_gradient(Ax, b, max_iter=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = Ax(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-10)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def kl_divergence(logits_old, logits_new):
    probs_old = torch.softmax(logits_old, dim=-1)
    probs_new = torch.softmax(logits_new, dim=-1)
    kl = torch.sum(probs_old * (torch.log(probs_old + 1e-10) - torch.log(probs_new + 1e-10)), dim=-1)
    return kl.mean()


def trpo_step(policy_net, states, actions, advantages, old_log_probs, max_kl=0.01, damping=0.1):
    # Compute the policy gradient
    logits = policy_net(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate_loss = (ratio * advantages).mean()

    # Compute the gradient of the surrogate loss
    grads = torch.autograd.grad(surrogate_loss, policy_net.parameters(), retain_graph=True)
    grads = torch.cat([grad.view(-1) for grad in grads])

    # Define the Fisher Information Matrix (FIM)
    def fisher_vector_product(v):
        kl = kl_divergence(logits.detach(), policy_net(states))
        grads_kl = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        grads_kl = torch.cat([g.view(-1) for g in grads_kl])
        kl_v = (grads_kl * v).sum()
        grads_kl_v = torch.autograd.grad(kl_v, policy_net.parameters())
        grads_kl_v = torch.cat([g.contiguous().view(-1) for g in grads_kl_v])
        return grads_kl_v + damping * v

    # Solve for the step direction
    step_dir = conjugate_gradient(fisher_vector_product, grads)

    # Compute the step size
    s = 0.5 * (grads @ step_dir)
    step_size = torch.sqrt(2 * max_kl / (s + 1e-10))

    # Apply the step
    with torch.no_grad():
        for param, step in zip(policy_net.parameters(), step_dir):
            param += step * step_size


# Main TRPO Implementation
if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='TRPO with SUMO simulation')
    parser.add_argument('--alpha_val', type=float, default=0.7, help='Reward parameter.')
    args = parser.parse_args()
    
    # Use the parsed alpha value
    alpha_val = args.alpha_val
    alpha = 0.01
    gamma = 0.99  # Discount factor
    runs = 25
    timesteps = 2000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    from my_maps import map_details
    for map_ in map_details:
        env = sumo_rl.env(
            net_file=map_['net_file'],
            route_file=map_['route_file'],
            use_gui=False,
            num_seconds=timesteps,
            reward_fn="weighted",
            fixed_ts=False,
            alpha = alpha_val
        )
        total_reward = 0
        for run in range(1, runs + 1):
            print(f"Starting Run {run} of {runs} on map {map_['net_file']}...")
            env.reset()
            policy_net = PolicyNetwork(state_size=env.observation_space(env.agents[0]).shape[0], action_size=env.action_space(env.agents[0]).n).to(device)
            value_net = ValueNetwork(state_size=env.observation_space(env.agents[0]).shape[0]).to(device)
            optimizer_value = torch.optim.Adam(value_net.parameters(), lr=alpha)

            states, actions, rewards, log_probs, dones = [], [], [], [], []

            for t in range(timesteps):
                for agent in env.agent_iter():
                    s, r, terminated, truncated, info = env.last()
                    done = terminated or truncated

                    state_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                    action, log_prob, _ = policy_net.get_action(state_tensor)

                    states.append(state_tensor)
                    actions.append(action)
                    rewards.append(r)
                    log_probs.append(log_prob)
                    dones.append(done)

                    total_reward += r

                    if done:
                        action = None
                    env.step(action)  # Perform the action in the environment

            # Save the model weights after each run
            save_dir = f"{alpha_val}/{map_['save_location']}trpo/"
            os.makedirs(save_dir, exist_ok=True)
            # torch.save(policy_net.state_dict(), f"{alpha_val}/models/trpo/{map_['net_file'].split('/')[-1][:map_['net_file'].split('/')[-1].index('.')]}/model_run_{run}.pth")
            # torch.save(value_net.state_dict(), f"{alpha_val}/models/trpo/{map_['net_file'].split('/')[-1][:map_['net_file'].split('/')[-1].index('.')]}/value_net_run_{run}.pth")

            # Save the CSV of agent rewards (if applicable)
            env.unwrapped.env.save_csv(f"{alpha_val}/{map_['save_location']}trpo/trpo", run)

            # Compute returns and advantages
            returns = []
            G = 0
            for r, done in zip(reversed(rewards), reversed(dones)):
                G = r + gamma * G * (1 - done)
                returns.insert(0, G)

            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            old_log_probs = torch.stack(log_probs).to(device)

            # Update value network
            values = value_net(states).squeeze()
            advantages = returns - values
            value_loss = nn.MSELoss()(values, returns)

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            # Update policy network using TRPO
            trpo_step(policy_net, states, actions, advantages, old_log_probs)
        env.close()
    print(total_reward)
