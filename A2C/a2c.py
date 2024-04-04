# Advantage Actor Critic
import torch
from torch import nn, optim
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.GELU(),
            nn.Linear(128,128),
            nn.GELU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        return self.network(input)


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.GELU(),
            nn.Linear(128,128),
            nn.GELU(),
            nn.Linear(128,1)
        )

    def forward(self, input: torch.Tensor):
        return self.network(input)


def main():
    env = gym.make("CartPole-v1")

    policy_net = Actor(env)
    value_net = Critic(env)

    gamma = 0.99

    policy_optim = optim.Adam(policy_net.parameters(), lr=1e-3, amsgrad=True)
    value_optim = optim.Adam(value_net.parameters(), lr=1e-3, amsgrad=True)

    max_episode_count = 1_000
    
    previous_total_rewards = []
    for episode in range(max_episode_count):

        state,_ = env.reset()

        done = False
        truncated = False
        total_reward = 0

        rewards = []
        logprobs = []
        values = []
        
        while not done and not truncated:
            probs = policy_net(torch.tensor(state))
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())

            state_value = value_net(torch.tensor(state))
            
            state = next_state

            values.append(state_value)
            rewards.append(reward)
            logprobs.append(-m.log_prob(action))

            total_reward += reward

        prev_discounted_return = 0
        all_returns = deque([])
        for t in range(len(rewards)-1,-1,-1):
            discounted_return = rewards[t] + gamma*prev_discounted_return
            prev_discounted_return = discounted_return
            all_returns.appendleft(discounted_return)
        
        values_pt = torch.stack(values).unsqueeze(1)
        discounted_returns_pt = torch.tensor(all_returns).unsqueeze(1)
        discounted_returns_pt = (discounted_returns_pt - discounted_returns_pt.mean()) / (discounted_returns_pt.std() + 1e-6)
        advantage = discounted_returns_pt - values_pt
        policy_loss = (torch.stack(logprobs).unsqueeze(1) * advantage.detach()).mean()
        value_loss = 0.5*advantage.square().mean()
        
        #combined = value_loss+ policy_loss

        value_optim.zero_grad()
        policy_optim.zero_grad()
        value_loss.backward()
        policy_loss.backward()
        #combined.backward()
        value_optim.step()
        policy_optim.step()

        previous_total_rewards.append(total_reward)
        
        if (len(previous_total_rewards) > 100):
            print(sum(previous_total_rewards[-100:])/100, episode, total_reward)



if (__name__ == "__main__"):
    main()
