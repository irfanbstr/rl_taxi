import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Explicitly set the backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create the Taxi environment with graphical rendering
env = gym.make("Taxi-v3", render_mode="rgb_array")

# Initialize Q-table and other parameters
state_space = env.observation_space.n
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 1000  # Reduce episodes for testing

# Track images for animation
frames = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = np.argmax(q_table[state])  # Choose the best action
        next_state, reward, done, truncated, info = env.step(action)

        # Update Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward
        step_count += 1

        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Episode {episode}, Step {step_count}, Total Reward: {total_reward}")
            # Capture frame every step to visualize
            frames.append(env.render())  # Capture frame for animation

    # Print progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# After training
print("Training finished!")

# Set up the plot for displaying animation
fig = plt.figure(figsize=(8, 8))
ims = []

for frame in frames:
    im = plt.imshow(frame)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
plt.show()
plt.pause(0.1)  # Ensure the window refreshes
