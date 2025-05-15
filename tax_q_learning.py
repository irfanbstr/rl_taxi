import gymnasium as gym
import numpy as np
import random
import time

# Create the Taxi environment
env = gym.make("Taxi-v3", render_mode="ansi")

# Initialize Q-table
state_space = env.observation_space.n
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

# Hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.95        # Discount factor
epsilon = 0.1      # Exploration rate
episodes = 10000   # Number of training episodes

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Exploration vs exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random action)
        else:
            action = np.argmax(q_table[state])  # Exploit (best known action)

        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Q-learning update rule
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value
        
        state = next_state  # Move to the next state

    # Optionally print every 1000 episodes
    if episode % 1000 == 0:
        print(f"Episode {episode}, Total reward: {total_reward}")

# After training, demonstrate the learned policy
state, _ = env.reset()
done = False
total_reward = 0

print("\nPolicy demonstration:\n")
while not done:
    action = np.argmax(q_table[state])  # Always exploit the learned policy
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    print(env.render())  # Print the environment state

print(f"\nTotal reward: {total_reward}")
env.close()
