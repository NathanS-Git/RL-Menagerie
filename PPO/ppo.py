import gymnasium as gym
import torch
from torch import nn,distributions,optim
from torch.distributions import Categorical
from collections import deque, namedtuple

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,env.action_space.n),
            nn.Softmax(-1)
        )

    def forward(self, input):
        return self.network(input)

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, input):
        return self.network(input)

def main():
    env = gym.make("CartPole-v1")

    policy_net = Actor(env)
    value_net = Critic(env)

    gamma = 0.99

    policy_optim = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
    value_optim = optim.AdamW(value_net.parameters(), lr=1e-4, amsgrad=True)

    max_episode_count = 1_000

    previous_total_rewards = []
    for episode in range(max_episode_count):
        state,_ = env.reset()

        done = False
        truncated = False
        total_reward = 0
    
        rewards = []
        old_probs = []
        states = [] 
        actions = []

        while not done and not truncated:
            probs = policy_net(torch.tensor(state))
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())

            states.append(torch.tensor(state))
            actions.append(action)
            rewards.append(reward)
            old_probs.append(probs[action].detach())
            
            state = next_state

            total_reward += reward

        prev_discounted_return = 0
        all_returns = deque([])
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + gamma*prev_discounted_return
            prev_discounted_return = discounted_return
            all_returns.appendleft(discounted_return)
        
        states_pt = torch.stack(states)

        for epoch in range(15):
            values_pt = value_net(states_pt).unsqueeze(1)
            discounted_returns_pt = torch.tensor(all_returns).unsqueeze(1)
            discounted_returns_pt = (discounted_returns_pt - discounted_returns_pt.mean()) / (discounted_returns_pt.std() + 1e-6)
            advantage = discounted_returns_pt - values_pt
            
            actions_pt = policy_net(states_pt).gather(1, torch.stack(actions).unsqueeze(1))
            p_diff = actions_pt / torch.stack(old_probs)
            
            epsilon = 0.2
            loss_clip = torch.min(p_diff*advantage, p_diff.clip(1-epsilon,1+epsilon)*advantage).mean()
            loss_vf = advantage.square().mean()
            loss = -loss_clip + loss_vf

            value_optim.zero_grad()
            policy_optim.zero_grad()
            loss.backward()
            value_optim.step()
            policy_optim.step()

        previous_total_rewards.append(total_reward)
        
        if (len(previous_total_rewards) > 100):
            print(sum(previous_total_rewards[-100:])/100, episode, total_reward)



if (__name__ == "__main__"):
    main()
