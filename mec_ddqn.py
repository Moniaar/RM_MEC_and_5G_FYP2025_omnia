class MECServer:
    def __init__(self, num_devices=10):
        self.global_model = build_cnn_model()
        self.devices = [EdgeDevice(i, cpu_freq=np.random.uniform(1e9, 3e9),
                                  energy=np.random.uniform(100, 500),
                                  bandwidth=np.random.uniform(10, 50)) 
                        for i in range(num_devices)]
        self.energy_history = []
        self.bandwidth_history = []

    def distribute_data(self):
        # Split CIFAR-10 data among devices (simplified)
        data_per_device = len(x_train) // len(self.devices)
        for i, device in enumerate(self.devices):
            start = i * data_per_device
            end = (i + 1) * data_per_device
            device.set_local_data(x_train[start:end], y_train[start:end])

    def fed_avg(self, local_weights, data_sizes):
        total_size = sum(data_sizes)
        avg_weights = [np.zeros_like(w) for w in local_weights[0]]
        for weights, size in zip(local_weights, data_sizes):
            for i in range(len(weights)):
                avg_weights[i] += weights[i] * size / total_size
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=1):
        local_weights = []
        data_sizes = []
        total_energy = 0
        total_bandwidth = 0
        
        for device_id in selected_devices:
            device = self.devices[device_id]
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            local_weights.append(weights)
            data_sizes.append(len(device.local_data[0]))
            total_energy += energy_used
            total_bandwidth += device.bandwidth
        
        # Aggregate models
        new_weights = self.fed_avg(local_weights, data_sizes)
        self.global_model.set_weights(new_weights)
        
        # Store metrics
        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)
        
        return T_local + T_trans  # Total delay

    def evaluate_global_model(self):
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=0)
        return accuracy

class DDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def select_devices_ddqn(mec_server, ddqn, num_devices_to_select):
    state = np.array([list(d.report_resources().values()) for d in mec_server.devices]).flatten()
    state = state.reshape(1, -1)
    selected = []
    for _ in range(num_devices_to_select):
        action = ddqn.act(state)
        if action not in selected:
            selected.append(action)
        # Simplified next state (in real case, update based on resource consumption)
        next_state = state
        reward = 1  # Simplified reward (could be based on energy/delay)
        ddqn.remember(state, action, reward, next_state, False)
        state = next_state
    return selected