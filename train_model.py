import pandas as pd

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Features used by the AI
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFailure counts:")
print(y.value_counts())
