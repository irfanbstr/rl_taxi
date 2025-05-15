import gymnasium as gym
import numpy as np
import time
import pandas as pd

# Load trained Q-table
q_table = np.load("data/q_table_exp1.npy")  # Make sure this path is correct

# Create Taxi environment with ANSI rendering
env = gym.make("Taxi-v3", render_mode="ansi")

episode_data = []

# Run multiple episodes
num_episodes = 1000
cumulative_reward = 0
total_stuck = 0

print("\n🚕 Policy Demonstration (ANSI View):\n")

for episode in range(num_episodes):
    state, _ = env.reset()
    initial_state = state 
    done = False
    total_reward = 0

    print(f"\n🚕 Episode {episode + 1}\n")
    max_steps = 50  # Avoid infinite loops
    done = False
    for step in range(max_steps):
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        #print(env.render())
        print(f"Action: {action}, Reward: {reward}, Total so far: {total_reward}\n")
        time.sleep(0.01)  # Adjust speed as needed

        if done:
            break
    else:    
        print("🚨 Max steps reached. Agent may be stuck.")
        total_reward = 0
        total_stuck +=1
        
     # Extract interpretable features from the initial state
    row, col, passenger_loc, destination = env.unwrapped.decode(initial_state)
    episode_data.append([row, col, passenger_loc, destination, total_reward])

    cumulative_reward += total_reward
    print(f"✅ Episode {episode + 1} Total Reward: {total_reward}\n")

# Final result
print(f"\n🎯 Cumulative Reward after {num_episodes} episodes: {cumulative_reward}")
print(f"Agent was stuck in {total_stuck} episodes.")
env.close()

df = pd.DataFrame(episode_data, columns=["row", "col", "passenger", "destination", "reward"])
df.to_csv("data/supervised_dataset_episodes.csv", index=False)