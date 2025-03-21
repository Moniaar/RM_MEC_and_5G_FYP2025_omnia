# Initialize MEC server and DDQN
mec_server = MECServer(num_devices=10)
mec_server.distribute_data()
ddqn = DDQN(state_size=30, action_size=10)  # 3 features per device * 10 devices

# Training loop
num_rounds = 20
for round_num in range(num_rounds):
    selected_devices = select_devices_ddqn(mec_server, ddqn, num_devices_to_select=5)
    delay = mec_server.simulate_training_round(selected_devices, epochs=1)
    accuracy = mec_server.evaluate_global_model()
    print(f"Round {round_num+1}: Accuracy = {accuracy:.4f}, Delay = {delay:.4f}")
    
    if len(ddqn.memory) > 32:
        ddqn.replay(32)
    ddqn.update_target_model()

# Plot energy and bandwidth graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mec_server.energy_history, label='Energy Consumption')
plt.xlabel('Training Round')
plt.ylabel('Energy Units')
plt.title('Energy Consumption Over Rounds')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mec_server.bandwidth_history, label='Bandwidth Usage')
plt.xlabel('Training Round')
plt.ylabel('Bandwidth (Mbps)')
plt.title('Bandwidth Usage Over Rounds')
plt.legend()

plt.tight_layout()
plt.show()