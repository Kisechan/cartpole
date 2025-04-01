import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 超参数
    EPISODES = 500
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.98

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = deque(maxlen=10000)

    epsilon = EPSILON_START
    rewards_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()  # 新API返回元组
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()

            # 新API返回5个值
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                next_states = torch.FloatTensor(np.array(next_states))
                actions = torch.LongTensor(np.array(actions))
                rewards = torch.FloatTensor(np.array(rewards))
                dones = torch.FloatTensor(np.array(dones))

                current_q = policy_net(states).gather(1, actions.unsqueeze(1))

                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}/{EPISODES}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), 'cartpole_dqn.pth')
    env.close()

    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training.png')
    plt.show()

def test_model():
    env = gym.make('CartPole-v1', render_mode='human')  # 添加渲染模式
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load('cartpole_dqn.pth'))
    model.eval()

    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            if done:
                print(f"Test Episode {episode+1}, Reward: {total_reward}")
                break

    env.close()

if __name__ == "__main__":
    print("==== Training Started ====")
    train_dqn()

    print("\n==== Testing Model ====")
    test_model()