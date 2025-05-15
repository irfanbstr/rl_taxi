import gymnasium as gym
import numpy as np
import time

# Load Q-table
q_table = np.load("data/q_table_exp1.npy")

# Create environment
env = gym.make("Taxi-v3", render_mode="ansi")

# Encode your custom input
start_state = env.unwrapped.encode(2, 3, 1, 0)  # Row=2, Col=3, Passenger=1, Destination=0

# Set the initial state
env.reset()
env.unwrapped.s = start_state

done = False
total_reward = 0

print("\n🚕 Custom Start State Policy Demonstration:\n")

while not done:
    action = np.argmax(q_table[env.unwrapped.s])
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward

    print(env.render())
    print(f"Action: {action}, Reward: {reward}, Total: {total_reward}\n")
    time.sleep(1.0)

print(f"\n🎯 Final Total Reward: {total_reward}")
env.close()
