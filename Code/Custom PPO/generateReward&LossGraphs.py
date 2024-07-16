import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = 'Dissertation/Dissertation/src/CustomPPO/training_logs.txt'

# Initialize lists to store the data
iterations = []
average_returns = []
average_losses = []

# Read the file and extract data
with open(file_path, 'r') as file:
    for line in file:
        if "Iteration" in line:
            iterations.append(int(line.split(":")[1].strip()))
        elif "Average Episodic Returns" in line:
            average_returns.append(float(line.split(":")[1].strip()))
        elif "Average actor loss" in line:
            average_losses.append(float(line.split(":")[1].strip()))

# Focus on the last 25% of the iterations
focus_ratio = 0.25
focus_start = int(len(iterations) * (1 - focus_ratio))

focused_iterations = iterations[focus_start:]
focused_average_returns = average_returns[focus_start:]
focused_average_losses = average_losses[focus_start:]

# Calculate the moving averages
window_size = 10
moving_average_returns = np.convolve(focused_average_returns, np.ones(window_size)/window_size, mode='valid')
moving_average_losses = np.convolve(focused_average_losses, np.ones(window_size)/window_size, mode='valid')

# Adjust iterations for the moving averages
adjusted_iterations = focused_iterations[window_size - 1:]

# Plotting the data
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plotting the average episodic returns
ax1.set_xlabel('Iteration', fontsize=14)
ax1.set_ylabel('Average Episodic Returns', fontsize=14, color='b')
ax1.plot(focused_iterations, focused_average_returns, linestyle='-', color='b', alpha=0.5, label='Average Episodic Returns')
ax1.plot(adjusted_iterations, moving_average_returns, linestyle='-', color='blue', label=f'Smoother Moving Average (window={window_size})')
ax1.tick_params(axis='y', labelcolor='b')

# Creating a second y-axis for the average actor loss
ax2 = ax1.twinx()
ax2.set_ylabel('Average Actor Loss', fontsize=14, color='r')
ax2.plot(focused_iterations, focused_average_losses, linestyle='-', color='r', alpha=0.5, label='Average Actor Loss')
ax2.plot(adjusted_iterations, moving_average_losses, linestyle='-', color='red', label=f'Smoother Moving Average (window={window_size})')
ax2.tick_params(axis='y', labelcolor='r')

# Adding titles and legends
fig.suptitle('Training Progress of PPO Algorithm (Focused on Later Iterations)', fontsize=16)
fig.tight_layout()  # To adjust the layout to make room for the title
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=12)

# Save the plot to a file
plt.savefig('ppo_training_progress_combined.png')

# Display the plot
plt.show()