# غيرنا النقطتين المن تيليغرام 4 و 7  لحد اخر مييت
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('miniset_confusion_matrix_sep26.txt', mode='a'),
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
GAMMA = 0.99
LR = 0.001
MIN_BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON = 0.1
EPSILON_MAX = 1.0
EPSILON_GROWTH = 1.001
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.99
TARGET_UPDATE = 10
NUM_EPISODES = 1000
NUM_DEVICES = 4
FEATURES_PER_DEVICE = 3
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.86
MAX_ITERATIONS = 1
ALPHA_N = 3.0
ALPHA_E = 2.0
ALPHA_L = 2.0
L_MAX = 500
G = 7000
TAU = 1e-28
MU_MB = 1
MU_BITS = MU_MB * 8 * 1e6
D = 20 * 1e6
LAMBDA = 1
ENERGY_SCALE = 1e12
SAMPLES_PER_DEVICE = 1276  # عدد العينات لكل جهاز (يجب أن يكون مضاعفًا لعدد الفئات)
SAMPLES_PER_CLASS = SAMPLES_PER_DEVICE // 10  # عدد العينات لكل فئة (1 عينة لكل فئة)

# CNN for DDQN
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Global Model for federated learning
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# FiveGNetwork
class FiveGNetwork:
    def __init__(self, base_bandwidth=1000, base_latency=0.005, capacity=1000, spectrum="sub6"):
        self.base_bandwidth = base_bandwidth
        self.base_latency = base_latency
        self.capacity = capacity
        self.connected_devices = {}
        self.spectrum = spectrum
        self.network_slice = {
            "bandwidth": base_bandwidth,
            "latency": base_latency,
        }
        self.interference_factor = 0.01
        self.fading_factor = 0.97 if spectrum == "sub6" else 0.9
        self.packet_loss_rate = 0.0003
        self.throughput_history = []
        self.latency_history = []
        self.packet_loss_history = []

    def connect_device(self, device_id):
        if len(self.connected_devices) >= self.capacity:
            logging.info(f"شبكة 5G: تم تجاوز السعة للجهاز {device_id}، يتم استخدام الخيار الاحتياطي.")
            return 10, 0.01, self.packet_loss_rate

        self.connected_devices[device_id] = True
        slice_info = self.network_slice
        num_devices = len(self.connected_devices)
        interference = self.interference_factor * (num_devices - 1)
        variation = np.random.uniform(0.85, 1.15) * self.fading_factor
        effective_bandwidth = slice_info["bandwidth"] * (1 - interference) * variation
        effective_bandwidth = max(5, effective_bandwidth)
        latency = np.random.uniform(0.003, 0.02) * (1 + 0.01 * num_devices)
        latency = min(latency, 0.005)
        packet_loss = self.packet_loss_rate * (1 + np.random.uniform(0.02, 0.1) * num_devices)
        if self.spectrum == "mmWave":
            packet_loss *= 1.3
            packet_loss = min(packet_loss, 0.05)

        self.throughput_history.append(effective_bandwidth)
        self.latency_history.append(latency)
        self.packet_loss_history.append(packet_loss)
        
        logging.info(f"الجهاز {device_id} متصل: النطاق الترددي={effective_bandwidth:.2f} Mbps، "
                     f"التأخير={latency*1000:.2f} ms، فقدان الحزم={packet_loss:.4f}، الخدمة=mMTC")
        
        return effective_bandwidth, latency, packet_loss

    def disconnect_device(self, device_id):
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            logging.info(f"الجهاز {device_id} تم فصله.")

    def reallocate_resources(self, selected_devices, devices):
        num_devices = len(self.connected_devices)
        num_active = len(selected_devices)
        if num_devices == 0:
            return
        active_bandwidth = self.base_bandwidth * 0.8 / max(1, num_active) if num_active > 0 else self.base_bandwidth
        inactive_bandwidth = self.base_bandwidth * 0.2 / max(1, num_devices - num_active) if num_devices > num_active else 0
        for device_id in self.connected_devices:
            original_bandwidth = self.network_slice["bandwidth"]
            self.network_slice["bandwidth"] = active_bandwidth if device_id in selected_devices else inactive_bandwidth
            effective_bandwidth, latency, packet_loss = self.connect_device(device_id)
            devices[device_id].update_network_params(effective_bandwidth, latency, packet_loss)
            self.network_slice["bandwidth"] = original_bandwidth
        logging.info(f"إعادة تخصيص الموارد: النطاق الترددي النشط={active_bandwidth:.2f} Mbps، النطاق الترددي غير النشط={inactive_bandwidth:.2f} Mbps")

    def simulate_channel_conditions(self):
        self.fading_factor = np.random.uniform(0.9, 0.98) if self.spectrum == "sub6" else np.random.uniform(0.85, 0.95)
        self.interference_factor = np.random.uniform(0.1, 0.005)
        logging.info(f"تم تحديث القناة: التلاشي={self.fading_factor:.2f}، التداخل={self.interference_factor:.2f}")

    def get_network_status(self):
        return {
            "connected_devices": len(self.connected_devices),
            "average_throughput": np.mean(self.throughput_history) if self.throughput_history else 0,
            "average_latency": np.mean(self.latency_history) if self.latency_history else self.base_latency,
            "average_packet_loss": np.mean(self.packet_loss_history) if self.packet_loss_history else self.packet_loss_rate,
            "total_bandwidth": self.network_slice["bandwidth"]
        }

# EdgeDevice
class EdgeDevice:
    def __init__(self, id, cpu_freq, energy, bandwidth, fiveg_network):
        self.id = id
        self.cpu_freq = cpu_freq
        self.energy = energy
        self.base_bandwidth = bandwidth
        self.fiveg_network = fiveg_network
        self.model = GlobalModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.local_data = None
        self.effective_bandwidth, self.latency, self.packet_loss = self.fiveg_network.connect_device(self.id)

    def update_network_params(self, effective_bandwidth, latency, packet_loss):
        self.effective_bandwidth = effective_bandwidth
        self.latency = latency
        self.packet_loss = packet_loss
        logging.info(f"الجهاز {self.id} تم تحديثه: النطاق الترددي={self.effective_bandwidth:.2f} Mbps، "
                     f"التأخير={self.latency*1000:.2f} ms، فقدان الحزم={self.packet_loss:.4f}")

    def set_local_data(self, data):
        self.local_data = data  # استبدال البيانات القديمة مباشرة
        logging.info(f"الجهاز {self.id}: تم تخصيص {len(data) if data else 0} عينة")

    def charge_energy(self, w=1):
        return np.random.poisson(w) * ENERGY_SCALE

    def train_local_model(self, epochs, energy_rate=1):
        if not hasattr(self, 'call_counter'):
            self.call_counter = 0
        self.call_counter += 1

        if self.local_data is None:
            logging.info(f"الجهاز {self.id}: لا توجد بيانات مخصصة، يتم تخطي التدريب.")
            return self.model.state_dict(), 0, 0, 0

        device = torch.device("cpu")
        self.model.to(device)
        self.model.train()

        batch_size = 1276
        loader = torch.utils.data.DataLoader(self.local_data, batch_size=batch_size, shuffle=True, drop_last=False)

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        T_local = MU_BITS * G / self.cpu_freq if self.cpu_freq > 0 else float('inf')
        B_k = (self.cpu_freq ** 2) * TAU * MU_BITS * G * ENERGY_SCALE
        C_k = self.charge_energy(energy_rate)
        logging.info(f"الجهاز {self.id}: الشحن بـ {C_k} وحدة طاقة.")
        if self.energy < B_k:
            logging.info(f"الجهاز {self.id}: الطاقة غير كافية ({self.energy:.2f} < {B_k:.2f})، يتم تخطي التدريب.")
            return self.model.state_dict(), 0, 0, 0
        self.energy = max(self.energy - B_k + C_k, 0)
        T_trans = D / self.effective_bandwidth if self.effective_bandwidth > 0 else float('inf')
        return self.model.state_dict(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.effective_bandwidth}

    def __del__(self):
        self.fiveg_network.disconnect_device(self.id)

# MECServer
class MECServer:
    def __init__(self, num_devices=NUM_DEVICES):
        self.fiveg_network = FiveGNetwork()
        self.global_model = GlobalModel()
        self.devices = []
        for i in range(num_devices):
            cpu_freq = np.random.uniform(0, 1)
            energy = np.random.uniform(2, 5)
            bandwidth = np.random.uniform(0, 2)
            device = EdgeDevice(i, cpu_freq, energy, bandwidth, self.fiveg_network)
            self.devices.append(device)
        self.energy_history = []
        self.bandwidth_history = []
        self.class_indices = {i: [] for i in range(10)}  # قوائم المؤشرات لكل فئة
        self.class_pointers = {i: 0 for i in range(10)}  # مؤشرات لتتبع العينات الموزعة لكل فئة
        self.distributed_indices = []  # لتتبع جميع المؤشرات الموزعة
        logging.info(f"تم إنشاء {len(self.devices)} أجهزة فريدة")

    def initialize_class_indices(self, trainset):
        """
        تهيئة قوائم المؤشرات لكل فئة بناءً على تسميات البيانات
        """
        for idx in range(len(trainset)):
            _, label = trainset[idx]
            self.class_indices[label].append(idx)
        for label in self.class_indices:
            random.shuffle(self.class_indices[label])  # خلط المؤشرات لكل فئة
            logging.info(f"الفئة {label}: {len(self.class_indices[label])} عينة")

    def distribute_data(self, trainset, selected_devices):
        """
        توزيع البيانات بشكل متوازن لتشمل جميع الفئات، مع استبدال البيانات القديمة.
        :param trainset: مجموعة البيانات التدريبية
        :param selected_devices: قائمة معرفات الأجهزة المختارة
        """
        # تهيئة قوائم المؤشرات لكل فئة إذا لم تكن مهيأة
        if not self.class_indices[0]:
            self.initialize_class_indices(trainset)

        for device_id in selected_devices:
            device = self.devices[device_id]
            device_indices = []

            # تخصيص عينات من كل فئة
            for label in range(10):  # الفئات من 0 إلى 9
                available_indices = [idx for idx in self.class_indices[label] if idx not in self.distributed_indices]
                if not available_indices:
                    # إعادة تعيين المؤشرات للفئة إذا نفدت
                    logging.warning(f"الفئة {label}: تم توزيع جميع العينات، إعادة تعيين المؤشرات")
                    available_indices = self.class_indices[label]
                    self.distributed_indices = [idx for idx in self.distributed_indices if idx not in available_indices]
                    self.class_pointers[label] = 0

                # اختيار SAMPLES_PER_CLASS عينة من الفئة الحالية
                num_samples = min(SAMPLES_PER_CLASS, len(available_indices) - self.class_pointers[label])
                if num_samples <= 0:
                    logging.warning(f"لا توجد عينات كافية للفئة {label} للجهاز {device_id}")
                    continue

                start_idx = self.class_pointers[label]
                end_idx = start_idx + num_samples
                selected_indices = available_indices[start_idx:end_idx]
                device_indices.extend(selected_indices)
                self.class_pointers[label] += num_samples
                self.distributed_indices.extend(selected_indices)

            # إنشاء مجموعة فرعية وتخصيصها للجهاز
            if device_indices:
                subset = torch.utils.data.Subset(trainset, device_indices)
                device.set_local_data(subset)
                # طباعة تفاصيل التوزيع
                print(f"تم توزيع البيانات على الجهاز {device_id}: المؤشرات {device_indices} (العدد الإجمالي: {len(device_indices)} عينة)")
            else:
                logging.warning(f"لا توجد بيانات موزعة للجهاز {device_id}")

        logging.info(f"تم توزيع {len(self.distributed_indices)} عينة حتى الآن")

    def fed_avg(self, local_weights):
        avg_weights = {key: torch.zeros_like(local_weights[0][key]) for key in local_weights[0]}
        num_clients = len(local_weights)
        for weights in local_weights:
            for key in weights:
                avg_weights[key] += weights[key] / num_clients
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=10):
        local_weights = []
        total_energy = 0
        total_bandwidth = 0
        delays = []
        logging.info(f"الأجهزة المختارة: {selected_devices}")
        self.fiveg_network.simulate_channel_conditions()
        self.fiveg_network.reallocate_resources(selected_devices, self.devices)

        # توزيع البيانات على الأجهزة المختارة
        self.distribute_data(trainset, selected_devices)

        for device_id in selected_devices:
            device = self.devices[device_id]
            total_bandwidth += device.effective_bandwidth
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            logging.info(f"الجهاز {device_id}: الطاقة المستخدمة: {energy_used} pJ، النطاق الترددي: {device.effective_bandwidth:.2f} Mbps")
            delay = T_local + T_trans
            delays.append(delay)
            if energy_used > 0:
                local_weights.append(weights)
                total_energy += energy_used

        if local_weights:
            new_weights = self.fed_avg(local_weights)
            self.global_model.load_state_dict(new_weights)
        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)
        max_latency = max(delays) if delays else 0
        return max_latency, total_energy, total_bandwidth, self.energy_history

    def evaluate_global_model(self, testloader):
        self.global_model.eval()
        correct = 0
        total = 0
        device = torch.device("cpu")
        self.global_model.to(device)
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        accuracy = correct / total
        return accuracy, all_preds, all_targets

# DeviceSelectionEnv
class DeviceSelectionEnv:
    def __init__(self, mec_server):
        self.mec_server = mec_server
        self.num_devices = len(mec_server.devices)
        self.e_max_per_device = np.array([device.energy for device in mec_server.devices])        
        self.e_max_total = np.sum(self.e_max_per_device)
        self.action_space = [i for i in range(self.num_devices)]

    def generate_state(self):
        state = np.array([list(device.report_resources().values()) for device in mec_server.devices])
        return state

    def step(self, action):
        selected_indices = [i for i, val in enumerate(action) if val == 1]
        num_selected = len(selected_indices)

        max_latency, total_energy_consumed, total_bandwidth, e = self.mec_server.simulate_training_round(selected_indices)
        self.e_max_total = np.sum(e)

        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy_consumed / self.e_max_total if self.e_max_total > 0 else 1.0)
                  - ALPHA_L * (max_latency / L_MAX))

        state = self.generate_state()
        logging.info(f"الجولة: المكافأة={reward:.2f}، الطاقة={total_energy_consumed}، النطاق الترددي={total_bandwidth:.2f}")
        return state, reward, max_latency

    def reset(self):
        return self.generate_state()

# ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, min_batch_size):
        return random.sample(self.memory, min_batch_size)

    def __len__(self):
        return len(self.memory)

# DDQNAgent
class DDQNAgent:
    def __init__(self, input_channels, output_dim):
        self.policy_net = CNN(input_channels, output_dim)
        self.target_net = CNN(input_channels, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def preprocess_state(self, state):
        state_array = np.array(state).flatten()
        target_size = 256
        padded_state = np.pad(state_array, (0, target_size - len(state_array)), mode='constant', constant_values=0)
        return padded_state.reshape(1, 16, 16)

    def select_action(self, state):
        n = len(state)
        action = [0] * n
        if random.random() < self.epsilon:
            num_selected = 1
            selected_indices = random.sample(range(n), num_selected)
            for idx in selected_indices:
                action[idx] = 1
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(self.preprocess_state(state), dtype=torch.float32).unsqueeze(0)
                scores = self.policy_net(state_tensor).squeeze()
                with open("scores_output.txt", "a") as f:
                    f.write(f"النتائج: {scores.tolist()}\n")
                num_selected = 1
                selected_indices = torch.argsort(scores, descending=True)[:num_selected].tolist()
                for idx in selected_indices:
                    action[idx] = 1
        return action

    def optimize_model(self):
        if len(self.memory) < MIN_BATCH_SIZE:
            return
        batch = self.memory.sample(MIN_BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.stack([self.preprocess_state(s) for s in states]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([self.preprocess_state(s) for s in next_states]), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        q_values = self.policy_net(states).squeeze()
        next_q_values_policy = self.policy_net(next_states).detach().squeeze()
        next_q_values_target = self.target_net(next_states).detach().squeeze()

        expected_q_values = []
        predicted_q_values = []
        for i in range(MIN_BATCH_SIZE):
            action_indices = [idx for idx, val in enumerate(actions[i]) if val == 1]
            if not action_indices:
                predicted_q_values.append(torch.tensor(0.0, dtype=torch.float32))
                expected_q_values.append(rewards[i])
                continue
            q_pred = q_values[i, action_indices]
            predicted_q_values.append(q_pred.mean())
            q_next = next_q_values_target[i, torch.argmax(next_q_values_policy[i])]
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

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('التنبؤ')
    plt.ylabel('الحقيقي')
    plt.title('مصفوفة الالتباس')
    plt.show()

# Main Simulation
if __name__ == "__main__":
    print("بدء البرنامج...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=85, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=85, shuffle=False, num_workers=2)

    mec_server = MECServer()
    env = DeviceSelectionEnv(mec_server)
    agent = DDQNAgent(1, NUM_DEVICES)
    rewards_per_episode = []
    episode_energy_history = []
    episode_bandwidth_history = []
    episode_accuracies = []
    iteration_accuracies = []
    episode_latency_history = []
    all_preds_list = []  # لتخزين التنبؤات لكل تكرار
    all_targets_list = []  # لتخزين الأهداف لكل تكرار

    for episode in range(NUM_EPISODES):
        logging.info(f"\nبدء الحلقة {episode+1}")
        state = env.reset()
        total_reward = 0
        iteration = 0
        avg_reward = float('inf')
        accuracy = 0.0
        max_accuracy = 0.0
        episode_energy = 0
        episode_bandwidth = 0
        episode_iterations = 0
        
        while accuracy < DESIRED_ACCURACY:
            action = agent.select_action(state)
            next_state, reward, max_latency = env.step(action)
            agent.memory.push((state, action, reward, next_state))
            agent.optimize_model()
            state = next_state
            total_reward += reward
            iteration += 1
            episode_iterations += 1
            episode_energy += mec_server.energy_history[-1] if mec_server.energy_history else 0
            episode_bandwidth += mec_server.bandwidth_history[-1] if mec_server.bandwidth_history else 0
            avg_reward = total_reward / iteration if iteration > 0 else reward

            accuracy, all_preds, all_targets = mec_server.evaluate_global_model(testloader)
            max_accuracy = max(max_accuracy, accuracy)
            iteration_accuracies.append(accuracy)
            all_preds_list.append(all_preds)
            all_targets_list.append(all_targets)
            logging.info(f"التكرار {iteration}، إجمالي المكافأة: {total_reward:.2f}، متوسط المكافأة: {avg_reward:.2f}, "
                        f"الدقة: {accuracy:.4f}، أقصى دقة: {max_accuracy:.2f}، الطاقة: {episode_energy:.2f}, "
                        f"النطاق الترددي: {episode_bandwidth:.2f}")

        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        episode_accuracies.append(max_accuracy)
        episode_energy_history.append(episode_energy / episode_iterations if episode_iterations > 0 else 0)
        episode_bandwidth_history.append(episode_bandwidth / episode_iterations if episode_iterations > 0 else 0)
        episode_latency_history.append(max_latency)
        logging.info(f"انتهت الحلقة {episode+1}، إجمالي المكافأة: {total_reward:.2f}، متوسط المكافأة: {avg_reward:.2f}, "
                    f"التكرارات: {iteration}، أقصى دقة: {max_accuracy:.2f}، الطاقة: {episode_energy_history[-1]:.2f}, "
                    f"النطاق الترددي: {episode_bandwidth_history[-1]:.2f}")
        
    logging.info(f"دقة الحلقات حتى الآن: {episode_accuracies}")
    total_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0
    logging.info(f"إجمالي الدقة عبر جميع الحلقات: {total_accuracy:.4f}")

    # رسم مصفوفة الالتباس باستخدام التنبؤات من التكرار 160
    if len(all_preds_list) >= 160 and len(all_targets_list) >= 160:
        classes = [str(i) for i in range(10)]  # فئات MNIST (0-9)
        plot_confusion_matrix(all_targets_list[159], all_preds_list[159], classes)
    else:
        logging.warning("لا يمكن رسم مصفوفة الالتباس: تم إكمال أقل من 160 تكرارًا")

    # رسم المقاييس الأخرى
    plt.figure(figsize=(10, 6))
    plt.plot(episode_energy_history, label="متوسط الطاقة لكل حلقة")
    plt.xlabel("الحلقة")
    plt.ylabel("الطاقة (pJ)")
    plt.title("استهلاك الطاقة")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episode_latency_history, label="التأخير لكل حلقة")
    plt.xlabel("الحلقة")
    plt.ylabel("التأخير (ثانية)")
    plt.title("التأخير لكل حلقة")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_accuracies, label="الدقة لكل تكرار")
    plt.xlabel("التكرار")
    plt.ylabel("الدقة")
    plt.title("دقة النموذج عبر الحلقات")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_accuracies[:160], label="الدقة لكل تكرار (أول 160)")
    plt.xlabel("التكرار")
    plt.ylabel("الدقة")
    plt.title("دقة النموذج خلال أول 160 تكرار")
    plt.legend()
    plt.show()
