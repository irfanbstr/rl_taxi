# exp_1_default.py
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_q_learning import train_q_learning

q_table, rewards = train_q_learning(
    alpha=1,
    gamma=0.1,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,  # Faster decay
    episodes=10000
)

np.save("data/q_table_exp2.npy", q_table)
np.save("data/rewards_exp2.npy", rewards)