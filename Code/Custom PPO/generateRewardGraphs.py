import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = 'Dissertation/Dissertation/src/CustomPPO/training_logs.txt'

# Initialize lists to store the data
iterations = []
average_returns = []

# Read the file and extract data
with open(file_path, 'r') as file:
    for line in file:
        if "Iteration" in line:
            iterations.append(int(line.split(":")[1].strip()))
        elif "Average Episodic Returns" in line:
            average_returns.append(float(line.split(":")[1].strip()))

# Focus on the last 25% of the iterations
focus_ratio = 0.25
focus_start = int(len(iterations) * (1 - focus_ratio))

focused_iterations = iterations[focus_start:]
focused_average_returns = average_returns[focus_start:]

# Calculate the moving average
window_size = 100
moving_average = np.convolve(focused_average_returns, np.ones(window_size)/window_size, mode='valid')

# Adjust iterations for the moving average
adjusted_iterations = focused_iterations[window_size - 1:]

# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(focused_iterations, focused_average_returns, linestyle='-', color='b', alpha=0.5, label='Average Episodic Rewards')
plt.plot(adjusted_iterations, moving_average, linestyle='-', color='r', label=f'Moving Average (window={window_size})')

# Adding titles and labels
plt.title('PPO Training Progress', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Average Episodic Rewards', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Adding a legend
plt.legend(fontsize=12)

# Save the plot to a file
plt.savefig('ppo_training_progress_rewards.png')

# Display the plot
plt.show()