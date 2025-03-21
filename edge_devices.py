class EdgeDevice:
    def __init__(self, id, cpu_freq, energy, bandwidth):
        self.id = id
        self.cpu_freq = cpu_freq  # CPU frequency in Hz
        self.energy = energy      # Energy units
        self.bandwidth = bandwidth  # Bandwidth in Mbps
        self.model = build_cnn_model()
        self.local_data = None

    def set_local_data(self, x_data, y_data):
        self.local_data = (x_data, y_data)

    def train_local_model(self, epochs):
        if self.local_data is None:
            raise ValueError("Local data not set!")
        x, y = self.local_data
        history = self.model.fit(x, y, epochs=epochs, verbose=0)
        
        # Calculate computational time and energy consumption
        G = 1e6  # CPU cycles per bit (example value)
        mu = len(x)  # Dataset size
        T_local = mu * G / self.cpu_freq  # Local training time
        tau = 1e-9  # Switched capacitance (example)
        B_k = (self.cpu_freq ** 2) * tau * mu * G  # Energy consumption
        self.energy -= B_k  # Update energy
        
        # Transmission time
        D = 1e6  # Model size in bits (example)
        T_trans = D / self.bandwidth  # Transmission time
        
        return self.model.get_weights(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.bandwidth}