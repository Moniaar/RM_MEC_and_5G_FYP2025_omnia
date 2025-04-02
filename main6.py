import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# تعريف شبكة CNN
class CNN(nn.Module):
    def __init__(self, input_channels, num_devices):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d((5, 2))  # التأكد من توافق الأبعاد
        self.fc1 = nn.Linear(32 * 5 * 2, num_devices)  # إخراج يتوافق مع عدد الأجهزة

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # تحويل المصفوفة إلى شكل متوافق مع الطبقة المتصلة
        x = self.fc1(x)
        return x

# تعريف خوارزمية DDQN
class DDQN:
    def __init__(self, input_channels, num_devices, lr=0.001, gamma=0.99):
        self.num_devices = num_devices
        self.model = CNN(input_channels, num_devices)
        self.target_model = CNN(input_channels, num_devices)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.memory = []

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.sample(range(self.num_devices), k=random.randint(1, self.num_devices))  # اختيار عشوائي
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state).unsqueeze(0))
                return torch.topk(q_values, k=random.randint(1, self.num_devices))[1].tolist()  # اختيار أفضل الأجهزة

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)  # الحفاظ على حجم الذاكرة محدودًا

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()
        
        target_q_values = rewards + self.gamma * next_q_values.max(1)[0] * (1 - dones)
        
        loss = self.criterion(q_values.max(1)[0], target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# دالة حساب المكافأة
def compute_reward(selected_devices, total_devices, energy_list, latency_list, alpha_n, alpha_e, alpha_l):
    m = len(selected_devices)  
    n = total_devices  

    if m == 0:  
        return 0  # لا يوجد أجهزة مختارة، المكافأة صفر

    E = sum(energy_list[i] for i in selected_devices)
    E_max = max(energy_list) if len(energy_list) > 0 else 1  # تصحيح الشرط

    L = max(latency_list[i] for i in selected_devices)
    L_max = max(latency_list) if len(latency_list) > 0 else 1  # تصحيح الشرط

    reward = (alpha_n * (m / n)) - (alpha_e * (E / E_max)) - (alpha_l * (L / L_max))
    return reward

# تنفيذ التدريب
def train_federated_learning(ddqn, episodes, batch_size, epsilon_decay, min_accuracy=0.85):
    epsilon = 1.0  
    accuracy = 0.0  

    for episode in range(episodes):
        state = np.random.rand(1, 1, 28, 28)  # حالة عشوائية تمثل بيانات الإدخال
        
        total_devices = 10  
        energy_list = np.random.rand(total_devices)  
        latency_list = np.random.rand(total_devices)  

        done = False
        steps = 0
        while not done:
            selected_devices = ddqn.select_action(state, epsilon)
            reward = compute_reward(selected_devices, total_devices, energy_list, latency_list, 1.0, 0.5, 0.5)
            
            next_state = np.random.rand(1, 1, 28, 28)  
            accuracy += 0.01  # محاكاة تحسن الدقة

            done = accuracy >= min_accuracy

            ddqn.store_experience(state, selected_devices, reward, next_state, done)
            state = next_state
            steps += 1

        ddqn.train(batch_size)
        ddqn.update_target_model()
        epsilon = max(0.1, epsilon * epsilon_decay)

        print(f"الحلقة {episode+1}: الدقة الحالية = {accuracy:.4f}, عدد الخطوات = {steps}, إبسيلون = {epsilon:.4f}")

# تنفيذ التجربة
input_channels = 1
num_devices = 10
ddqn = DDQN(input_channels, num_devices)

train_federated_learning(ddqn, episodes=100, batch_size=32, epsilon_decay=0.99)
