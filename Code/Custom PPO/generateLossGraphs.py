import matplotlib.pyplot as plt
import numpy as np

# Define the file path
file_path = 'Dissertation/Dissertation/src/CustomPPO/training_logs.txt'

# Initialize lists to store the data
iterations = []
average_losses = []

# Read the file and extract data
with open(file_path, 'r') as file:
    for line in file:
        if "Iteration" in line:
            iterations.append(int(line.split(":")[1].strip()))
        elif "Average actor loss" in line:
            average_losses.append(float(line.split(":")[1].strip()))

# Focus on the last 25% of the iterations
focus_ratio = 0.25
focus_start = int(len(iterations) * (1 - focus_ratio))

focused_iterations = iterations[focus_start:]
focused_average_losses = average_losses[focus_start:]

# Calculate the moving average
window_size = 10
moving_average = np.convolve(focused_average_losses, np.ones(window_size)/window_size, mode='valid')

# Adjust iterations for the moving average
adjusted_iterations = focused_iterations[window_size - 1:]

# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(focused_iterations, focused_average_losses, linestyle='-', color='b', alpha=0.5, label='Average Actor Loss')
plt.plot(adjusted_iterations, moving_average, linestyle='-', color='r', label=f'Smoother Moving Average (window={window_size})')

# Adding titles and labels
plt.title('Training Progress of PPO Algorithm (Actor Loss Focused on Later Iterations)', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Average Actor Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Adding a legend
plt.legend(fontsize=12)

# Save the plot to a file
plt.savefig('ppo_training_loss_progress.png')

# Display the plot
plt.show()
