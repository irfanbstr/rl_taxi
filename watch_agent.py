import gymnasium as gym
import numpy as np
import time

# Load trained Q-table
q_table = np.load("data/q_table_exp2.npy")  # Update path if needed

# Create Taxi environment with ANSI rendering
env = gym.make("Taxi-v3", render_mode="ansi")
state, _ = env.reset()
done = False
total_reward = 0

print("\n🚕 Policy Demonstration (ANSI View):\n")

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # Print rendered environment to terminal
    print(env.render())
    print(f"Action taken: {action}, Reward: {reward}, Total so far: {total_reward}\n")
    time.sleep(1.0)

print(f"\n✅ Finished with Total Reward: {total_reward}")
env.close()