# Advantage Actor Critic
from re import A
from numpy import format_float_scientific
import torch
from torch import nn, optim
from torch.distributions import Categorical
import gymnasium as gym

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

    policy_optim = optim.SGD(policy_net.parameters(), lr=1e-4)
    value_optim = optim.SGD(value_net.parameters(), lr=1e-4)

    max_episode_count = 1_000
    
    previous_total_rewards = []
    for episode in range(max_episode_count):

        state,_ = env.reset()

        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            probs = policy_net(torch.tensor(state))
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())

            value = value_net(torch.tensor(state))
            advantage = reward + gamma*value_net(torch.tensor(next_state))*(not done) - value
            state = next_state

            policy_loss = -m.log_prob(action) * advantage
            combined = policy_loss + 0.5*advantage.square()
            policy_optim.zero_grad()
            value_optim.zero_grad()
            combined.backward()
            #for p in policy_net.parameters():
            #    p.grad *= -advantage
            policy_optim.step()

            value_optim.step()

            total_reward += reward

        previous_total_rewards.append(total_reward)
        
        if (len(previous_total_rewards) > 100):
            print(sum(previous_total_rewards[-100:])/100, episode, total_reward)



if (__name__ == "__main__"):
    main()
