import numpy as np
import matplotlib.pyplot as plt


rewards = np.load("data/rewards_exp2.npy")
plt.plot(rewards)
plt.title("Reward over Episodes")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()