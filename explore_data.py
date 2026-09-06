import pandas as pd

df = pd.read_csv("ai4i2020.csv")

print("=== DATASET INFO ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n=== FAILURE COUNTS ===")
print(df["Machine failure"].value_counts())

print("\n=== FAILURE PERCENTAGE ===")
print(df["Machine failure"].value_counts(normalize=True) * 100)

print("\n=== AVERAGE VALUES ===")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print(df.groupby("Machine failure")[features].mean())

print("\n=== FAILURE TYPES ===")

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

for col in failure_types:
    print(col, ":", df[col].sum())
