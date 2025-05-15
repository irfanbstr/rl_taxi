import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/reward_predictor.pkl")

# Predict reward for a sample state
sample_input = pd.DataFrame([[2, 3, 1, 0]], columns=["row", "col", "passenger", "destination"])
predicted_reward = model.predict(sample_input)
print(f"Estimated reward: {predicted_reward[0]}")