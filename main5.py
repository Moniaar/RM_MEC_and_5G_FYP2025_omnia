import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# تحديث المعاملات بناءً على الورقة العلمية
GAMMA = 0.99  # عامل التخفيض
LR = 0.0005  # معدل التعلم
BATCH_SIZE = 32  # حجم الدفعة التدريبية
MEMORY_SIZE = 20000  # حجم الذاكرة
EPSILON = 0.9  # معدل الاستكشاف الابتدائي
EPSILON_MIN = 0.01  # الحد الأدنى لـ Epsilon
EPSILON_DECAY = 0.996  # معدل التناقص التدريجي لـ Epsilon
TARGET_UPDATE = 10  # عدد الحلقات قبل تحديث الشبكة الهدف
NUM_EPISODES = 200  # عدد الحلقات التدريبية
NUM_DEVICES = 20  # عدد الأجهزة
NUM_SELECTED_DEVICES = 5  # عدد الأجهزة المختارة
FEATURES_PER_DEVICE = 3  # الميزات لكل جهاز
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)

ALPHA_N = 3.0  # معامل عدد الأجهزة المختارة
ALPHA_E = 2.0  # معامل استهلاك الطاقة
ALPHA_L = 2.0  # معامل التأخير
E_MAX = 100  # الحد الأقصى لاستهلاك الطاقة
L_MAX = 500  # الحد الأقصى للتأخير ثانية
TARGET_ACCURACY = 0.95  # الدقة المطلوبة لإنهاء الحلقة

# نموذج CNN لاختيار الأجهزة
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))  # تصغير الأبعاد للحجم المطلوب
        self.fc1 = nn.Linear(32 * 4 * 2, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# بيئة اختيار الأجهزة
class DeviceSelectionEnv:
    def __init__(self, num_devices=NUM_DEVICES):
        self.num_devices = num_devices
        self.state = self.generate_state()
        self.accuracy = 0.0

    def generate_state(self):
        latency = np.random.rand(self.num_devices, 1) * 10
        bandwidth = np.random.rand(self.num_devices, 1) * 100
        battery = np.random.randint(10, 101, (self.num_devices, 1))
        return np.hstack((latency, bandwidth, battery))

    def step(self, action):
        selected_devices = np.array(self.state)[action]
        num_selected = len(action)
        total_energy = np.sum(selected_devices[:, 2])
        max_latency = np.max(selected_devices[:, 0])
        self.accuracy = min(1.0, self.accuracy + 0.02)
        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy / E_MAX)
                  - ALPHA_L * (max_latency / L_MAX)
                  + 10 * (self.accuracy - 0.5))
        self.state = self.generate_state()
        return self.state, reward, self.accuracy >= TARGET_ACCURACY

    def reset(self):
        self.state = self.generate_state()
        self.accuracy = 0.0
        return self.state

# ذاكرة إعادة التشغيل
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# وكيل DDQN
class DDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = CNN(1, output_dim)
        self.target_net = CNN(1, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.sample(range(len(state)), NUM_SELECTED_DEVICES)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state).reshape(1, 1, NUM_DEVICES, FEATURES_PER_DEVICE), dtype=torch.float32)
                scores = self.policy_net(state_tensor).squeeze()
                return torch.argsort(scores, descending=True)[:NUM_SELECTED_DEVICES].tolist()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states).reshape(-1, 1, NUM_DEVICES, FEATURES_PER_DEVICE), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states).reshape(-1, 1, NUM_DEVICES, FEATURES_PER_DEVICE), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_q_values_policy = self.policy_net(next_states).detach()
        best_actions = torch.argmax(next_q_values_policy, dim=1)
        next_q_values_target = self.target_net(next_states).detach()
        selected_q_values = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        expected_q_values = rewards + GAMMA * selected_q_values
        loss = nn.MSELoss()(self.policy_net(states).sum(dim=1), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# تنفيذ الحلقة التدريبية
env = DeviceSelectionEnv()
agent = DDQNAgent(INPUT_DIM, NUM_DEVICES)
rewards_per_episode = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.memory.push((state, action, reward, next_state))
        agent.optimize_model()
        state = next_state
        total_reward += reward
    
    agent.update_epsilon()
    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()
    
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Selected Devices: {action}")

torch.save(agent.policy_net.state_dict(), "ddqn_model.pth")
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
