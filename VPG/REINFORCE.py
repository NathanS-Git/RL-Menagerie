from re import A
from typing import Any
import torch
from torch import nn, optim
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque, namedtuple


device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
Transition = namedtuple('Transition', ('reward', 'logprob'))

class REINFORCE(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        return self.network(input)


def main():
    env = gym.make("CartPole-v1")

    policy_net = REINFORCE(env).to(device)

    max_episode_count = 1_000

    gamma = 0.99
    lr = 1e-3

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    
    previous_total_rewards = []
    for episode in range(max_episode_count):
        state,_ = env.reset()
        episode_history = deque([])
        
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            probs = policy_net(torch.tensor(state).to(device))
            m = Categorical(probs)
            action = m.sample()
            
            state, reward, done, truncated, _ = env.step(action.item())
            total_reward += reward
            episode_history.append(Transition(reward, m.log_prob(action)))

        previous_total_rewards.append(total_reward)

        L = torch.tensor(0, dtype=torch.float32).to(device)
        prev_discounted_return = 0
        all_returns = deque([])
        for t in range(len(episode_history)-1,-1,-1):
            discounted_return = episode_history[t].reward + gamma*prev_discounted_return
            prev_discounted_return = discounted_return
            all_returns.appendleft(discounted_return)

        returns = torch.tensor(all_returns)
        # Normalize returns to help stabilize
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for t in range(len(episode_history)):
            L += returns[t] * -episode_history[t].logprob
        L /= len(episode_history)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()
            
        if (len(previous_total_rewards) > 100):
            print(sum(previous_total_rewards[-100:])/100,episode,total_reward)
            


if (__name__ == "__main__"):
    main()
