import gymnasium as gym
import numpy as np
import random

def train_q_learning(
    alpha=0.1,
    gamma=0.6,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    episodes=10000,
    max_steps=100,
    render=False,
    render_mode="ansi",
    seed=None,
):
    """
    Trains a Q-learning agent on the Taxi-v3 environment.

    Returns:
        q_table (np.ndarray): The learned Q-table
        rewards (list): List of total rewards per episode
    """
    env = gym.make("Taxi-v3", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)

    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = np.zeros((state_space, action_space))
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_steps:
            # Epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, info = env.step(action)

            # Q-learning update
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value

            state = next_state
            total_reward += reward
            step += 1

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)

        # Optional: print progress
        if episode % 1000 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    env.close()
    return q_table, rewards
