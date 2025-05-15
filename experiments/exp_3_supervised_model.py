import numpy as np
import gymnasium as gym
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Load Q-table
q_table = np.load("data/q_table_exp1.npy")

# Simulate data
env = gym.make("Taxi-v3")
X = []
y = []

for _ in range(5000):  # Generate 5000 samples
    state, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)

        X.append([state, action])
        y.append(reward)

        state = next_state
        steps += 1

env.close()

# Train supervised model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor().fit(X_train, y_train)
predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions))
