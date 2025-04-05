import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.serialization import INT_SIZE

# تحديث المعاملات بناءً على الورقة العلمية
GAMMA = 0.99  # عامل التخفيض
LR = 0.0005  # معدل التعلم
BATCH_SIZE = 32  # حجم الدفعة التدريبية
MEMORY_SIZE = 20000  # حجم الذاكرة
EPSILON = 0.1  # معدل الاستكشاف الابتدائي (ε-greedy: 0.1 → 1)
EPSILON_MAX = 1.0  # الحد الأقصى لـ Epsilon
EPSILON_GROWTH = 1.001  # معدل الزيادة التدريجي لـ Epsilon
TARGET_UPDATE = 10  # عدد الحلقات قبل تحديث الشبكة الهدف
NUM_EPISODES = 20  # عدد جولات التدريب العالمية (G = 7000)
NUM_DEVICES = 10  # عدد الأجهزة (10 IoT Devices)
NUM_SELECTED_DEVICES = 4  # عدد الأجهزة المختارة (η = 4)
FEATURES_PER_DEVICE = 3  # الميزات لكل جهاز (Latency, Bandwidth, Energy Consumption)
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.8  # دقة افتراضية للتوقف (يمكن تعديلها)

ALPHA_N = 3.0  # معامل عدد الأجهزة المختارة (α_n = 3)
ALPHA_E = 2.0  # معامل استهلاك الطاقة (α_e = 2)
ALPHA_L = 2.0  # معامل التأخير (α_l = 2)
E_MAX = 100  # الحد الأقصى لاستهلاك الطاقة (E_max = 100)
L_MAX = 500  # الحد الأقصى للتأخير (L_max = 500 s)

# نموذج CNN لاختيار الأجهزة (DDQN) وللنموذج العالمي
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(2, 2), padding=1)  # 2×2 Conv Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 2), padding=1)  # 2×2 Conv Layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))  # 2×2 Pooling Layer
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# نموذج عالمي بسيط لمحاكاة التعلم الاتحادي (Global Model)
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Add this part
        dummy_input = torch.zeros(1, 1, 28, 28)
        dummy_out = self.pool(torch.relu(self.conv1(dummy_input)))
        flat_size = dummy_out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# بيئة اختيار الأجهزة
class DeviceSelectionEnv:
    def __init__(self, num_devices=NUM_DEVICES):
        super(DeviceSelectionEnv, self).__init__()
        self.num_devices = num_devices
        self.state = self.generate_state()
        self.action_space = [i for i in range(num_devices)]

    def generate_state(self):
        # توليد قيم واقعية بناءً على الورقة
        latency = np.random.rand(self.num_devices, 1) * L_MAX  # التأخير بين 0 و L_max (500 ثانية)
        bandwidth = np.random.rand(self.num_devices, 1) * 2  # عرض النطاق بين 0 و 2 ميجابت/ثانية (R = 2 Mbps)
        energy = np.random.rand(self.num_devices, 1) * E_MAX  # استهلاك الطاقة بين 0 و E_max (100)
        return np.hstack((latency, bandwidth, energy))

    def step(self, action):
        selected_devices = np.array(self.state)[action]
        num_selected = len(action)  # m: عدد الأجهزة المختارة
        total_energy = np.sum(selected_devices[:, 2])  # E: مجموع استهلاك الطاقة
        max_latency = np.max(selected_devices[:, 0])  # L: أقصى تأخير

        # معادلة المكافأة بناءً على الورقة (Equation 13):
        # R(s, a) = α_n * (m/n) - α_e * (E/E_max) - α_l * (L/L_max)
        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy / E_MAX)
                  - ALPHA_L * (max_latency / L_MAX))

        self.state = self.generate_state()
        return self.state, reward, False

    def reset(self):
        self.state = self.generate_state()
        return self.state

# ذاكرة إعادة التشغيل
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def len(self):
        return len(self.memory)

# وكيل DDQN لاختيار العملاء
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
        if self.memory.len() < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states).reshape(-1, 1, NUM_DEVICES, FEATURES_PER_DEVICE), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states).reshape(-1, 1, NUM_DEVICES, FEATURES_PER_DEVICE), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Predict Q-values for current and next states
        q_values = self.policy_net(states)             # (BATCH_SIZE, NUM_DEVICES)
        next_q_values_policy = self.policy_net(next_states).detach()
        next_q_values_target = self.target_net(next_states).detach()

        expected_q_values = []
        predicted_q_values = []

        for i in range(BATCH_SIZE):
            action_indices = actions[i]
            q_pred = q_values[i][action_indices]  # Current Q-values of selected actions
            q_next = next_q_values_target[i][torch.argmax(next_q_values_policy[i])]  # Double DQN target

            # Average current Q-values of selected devices
            predicted_q_values.append(q_pred.mean())
            expected_q_values.append(rewards[i] + GAMMA * q_next)

        predicted_q_values = torch.stack(predicted_q_values)
        expected_q_values = torch.stack(expected_q_values)

        loss = nn.MSELoss()(predicted_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_epsilon(self):
        self.epsilon = min(EPSILON_MAX, self.epsilon * EPSILON_GROWTH)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# محاكاة التدريب المحلي لكل عميل
def local_training(model, num_epochs=5):  # E = 5
    dummy_input = torch.randn(32, 1, 28, 28)  # Adjust based on expected input shape
    output = model(dummy_input)
    print("Model Output Shape:", output)

    dummy_labels = torch.randint(0, 10, (32,))  # تسميات وهمية
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        print("Outputs:", outputs)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
    return model.state_dict()

# تجميع الأوزان باستخدام FedAvg
def aggregate_weights(global_model, local_weights):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.mean(torch.stack([weights[key] for weights in local_weights]), dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

# تنفيذ الحلقة التدريبية
env = DeviceSelectionEnv()
agent = DDQNAgent(INPUT_DIM, NUM_DEVICES)  # تهيئة وكيل DDQN
global_model = GlobalModel()  # تهيئة النموذج العالمي
rewards_per_episode = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    iteration = 0
    avg_reward = -float('inf')

    # التكرار حتى تحقيق الدقة المطلوبة
    # while avg_reward < DESIRED_ACCURACY and iteration < 100:
    while iteration < 5:  # تعديل الشرط للتكرار
        action = agent.select_action(state)  # اختيار العملاء باستخدام DDQN
        next_state, reward, _ = env.step(action)
        agent.memory.push((state, action, reward, next_state))
        agent.optimize_model()
        state = next_state
        total_reward += reward
        iteration += 1
        avg_reward = total_reward / iteration if iteration > 0 else reward


# محاكاة التدريب المحلي للعملاء المختارين
        local_weights = []
        for client_id in action:
            local_model = GlobalModel()
            local_model.load_state_dict(global_model.state_dict())  # نسخ النموذج العالمي
            local_weights.append(local_training(local_model))  # تدريب محلي

        # تحديث النموذج العالمي باستخدام FedAvg
        global_model = aggregate_weights(global_model, local_weights)

    agent.update_epsilon()
    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()

    rewards_per_episode.append(total_reward)
    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Iterations: {iteration}, Selected Devices: {action}")

# حفظ النموذج
torch.save(agent.policy_net.state_dict(), "ddqn_model.pth")
torch.save(global_model.state_dict(), "global_model.pth")

# رسم الأداء التدريبي
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance (DDQN with CNN in Federated Learning)")
plt.legend()
plt.show()
