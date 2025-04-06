# This code is based on a dummy input and not on our database (CIFAR-10)
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
MIN_BATCH_SIZE = 32  # حجم الدفعة التدريبية
MEMORY_SIZE = 20000  # حجم الذاكرة
EPSILON = 0.1  # معدل الاستكشاف الابتدائي (ε-greedy: 0.1 → 1)
EPSILON_MAX = 1.0  # الحد الأقصى لـ Epsilon
EPSILON_GROWTH = 1.001  # معدل الزيادة التدريجي لـ Epsilon
EPSILON_MIN = 0.0  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay factor (adjustable)
TARGET_UPDATE = 10  # عدد الحلقات قبل تحديث الشبكة الهدف
NUM_EPISODES = 10  # عدد جولات التدريب العالمية (G = 7000)
NUM_DEVICES = 4  # عدد الأجهزة (10 IoT Devices)
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
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))  # 16×16 → 8×8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))  # 8×8 → 4×4
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # 64 channels × 4 × 4 = 1024
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)  # First pooling
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)  # Second pooling
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 1024]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# نموذج عالمي بسيط لمحاكاة التعلم الاتحادي (Global Model) this will be edited when we have the dataset
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
        # values from the paper we need to add and configure
        latency = np.random.rand(self.num_devices, 1) * L_MAX  # التأخير بين 0 و L_max (500 ثانية)
        bandwidth = np.random.rand(self.num_devices, 1) * 2  # عرض النطاق بين 0 و 2 ميجابت/ثانية (R = 2 Mbps)
        energy = np.random.rand(self.num_devices, 1) * E_MAX  # استهلاك الطاقة بين 0 و E_max (100)
        return np.hstack((latency, bandwidth, energy))

    def step(self, action):
        selected_devices = np.array(self.state)[action]
        num_selected = len(action)  # m: عدد الأجهزة المختارة
        # These 2 below needs to get changed into the correct formula according to the paper equations
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

    def sample(self, min_batch_size):
        return random.sample(self.memory, min_batch_size)

    def __len__(self):
        return len(self.memory)

# وكيل DDQN لاختيار العملاء
class DDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = CNN(1, output_dim)
        self.target_net = CNN(1, output_dim)
        # Adaptive Moment Estimation is an algorithm
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def preprocess_state(self, state):
        # Convert 10×3 state to 16×16 tensor
        state_array = np.array(state)  # Shape: [N, 3] where N could be 10 or 4
        flat_state = state_array.flatten()  # Shape: [N * 3]
        target_size = 256  # 16 × 16
        padded_state = np.pad(flat_state, (0, target_size - len(flat_state)), mode='constant', constant_values=0)
        # 1 Batch size (1 sample at a time)
        # 1 Number of channels (grayscale "image")
        # 16 Height and 16 Width (16×16 image)
        return padded_state.reshape(1, 1, 16, 16)  # Shape: [1, 1, 16, 16]
        # Flatten to 30 elements and pad to 256 (16×16)
        # CNN is used to taking the input as 1×16×16

# If a random number (between 0 and 1) is less than self.epsilon, the agent explores an action
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.sample(range(len(state)), NUM_SELECTED_DEVICES)
        else:
            # no_grad is used to reduce memory consumption for computations
            with torch.no_grad():
                # Converts the preprocessed state into a PyTorch tensor suitable for input to the neural networ
                state_tensor = torch.tensor(self.preprocess_state(state), dtype=torch.float32)
                # Passes the state_tensor through the policy network to get Q-values (or scores) for all possible
                # actions and removes unnecessary dimensions from the output.
                # the squeeze function Removes dimensions of size 1 from the tensor. Here, it converts the [1, 10]
                scores = self.policy_net(state_tensor).squeeze()
                return torch.argsort(scores, descending=True)[:NUM_SELECTED_DEVICES].tolist()

    def optimize_model(self):
        if len(self.memory) < MIN_BATCH_SIZE:
            return

        batch = self.memory.sample(MIN_BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
# Converts the sampled states and next_states into tensors of shape
# [32, 1, 16, 16] using self.preprocess_state, and converts rewards into a tensor
        states = torch.tensor([self.preprocess_state(s) for s in states], dtype=torch.float32)
        next_states = torch.tensor([self.preprocess_state(s) for s in next_states], dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Predict Q-values
        q_values = self.policy_net(states)  # Shape: [MIN_BATCH_SIZE, NUM_DEVICES]
        next_q_values_policy = self.policy_net(next_states).detach()
        next_q_values_target = self.target_net(next_states).detach()

        expected_q_values = []
        predicted_q_values = []

        for i in range(MIN_BATCH_SIZE):
            action_indices = actions[i]
            q_pred = q_values[i][action_indices]  # Q-values of selected actions
            # Bellman Equation lies here
            q_next = next_q_values_target[i][torch.argmax(next_q_values_policy[i])]  # Double DQN target
            predicted_q_values.append(q_pred.mean())
            expected_q_values.append(rewards[i] + GAMMA * q_next)

        predicted_q_values = torch.stack(predicted_q_values)
        expected_q_values = torch.stack(expected_q_values)

        loss = nn.MSELoss()(predicted_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

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
    while iteration < DESIRED_ACCURACY:  # Loop until desired accuracy or max iterations
        # Line 5: ε-greedy action selection
        # Line 5: ε-greedy action selection
        if random.random() < EPSILON:  # With probability ε, select random action
            action = random.sample(env.action_space, NUM_SELECTED_DEVICES)  # Select 4 random devices
        else:  # With probability (1 - ε), select action that maximizes Q(s, a; θ)
            action = agent.select_action(state)  # Assuming this uses the Q-network

        next_state, reward, _ = env.step(action)
        # We need to add epsilon greedy exploration here
        agent.optimize_model()
        # observe new state and reward
        state = next_state
        total_reward += reward
        iteration += 1
        avg_reward = total_reward / iteration if iteration > 0 else reward
        # Store experience in memory
        agent.memory.push((state, action, reward, next_state))
# Line 8: Sample a minibatch of experiences from memory
        if len(agent.memory) >= MIN_BATCH_SIZE:  # Ensure enough experiences in memory
            experiences = agent.memory.sample(MIN_BATCH_SIZE)  # Sample m experiences
            states, actions, rewards, next_states = zip(*experiences)

            # Convert to appropriate format (e.g., numpy arrays or tensors)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)

# Line 9: Update θ with gradient descent using loss (Equation 18)
            # Compute target y (Equation 19)
            # y = R(s, a) + γ * Q'(s', argmax_a' Q(s', a'; θ); θ')
            q_values_next = agent.model(next_states)  # Q(s', a'; θ)
            best_actions = np.argmax(q_values_next, axis=1)  # argmax_a' Q(s', a'; θ)
            target_q_values = agent.target_model(next_states)  # Q'(s', a'; θ')
            targets = rewards + GAMMA * target_q_values[np.arange(MIN_BATCH_SIZE), best_actions]

# Compute current Q-values and loss
            q_values = agent.model(states)
            q_values[np.arange(MIN_BATCH_SIZE), actions] = targets  # Update with target values

            # Optimize model (gradient descent step)
            #The loss is implicitly calculated inside agent.optimize_model(states, q_values)
            agent.optimize_model(states, q_values)  # Assuming this computes loss and updates θ

# Line 10: Regularly reset θ' = θ (update target network)
        if iteration % TARGET_UPDATE == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())  # Copy weights to target

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
