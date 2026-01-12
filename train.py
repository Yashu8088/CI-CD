import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Load dataset
df = pd.read_csv(URL, sep=";")

# Convert to binary classification
df["quality"] = (df["quality"] >= 6).astype(int)

X = df.drop("quality", axis=1)
y = df["quality"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained using UCI data")
