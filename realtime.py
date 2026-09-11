import joblib
import pandas as pd
import time

# Load trained model
model = joblib.load("machine_failure_model.pkl")

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Features used by our model
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("🚀 REAL-TIME MACHINE MONITOR STARTED")
print("--------------------------------------")

# Simulate live sensor readings
for index, row in df.iterrows():

    # Get current machine readings
    data = pd.DataFrame([row[features].to_dict()])

    # Predict failure probability
    probability = model.predict_proba(data)[0][1]
    percentage = probability * 100

    # Determine risk
    if percentage < 30:
        status = "🟢 NORMAL"
    elif percentage < 70:
        status = "🟡 WARNING"
    else:
        status = "🔴 CRITICAL"

    # Display
    print(f"\nReading #{index + 1}")
    print(f"Temperature: {row['Air temperature [K]']:.1f} K")
    print(f"Speed: {row['Rotational speed [rpm]']:.0f} RPM")
    print(f"Torque: {row['Torque [Nm]']:.1f} Nm")
    print(f"Tool Wear: {row['Tool wear [min]']:.0f} min")
    print(f"Failure Risk: {percentage:.2f}%")
    print(f"Status: {status}")

    # Wait 1 second before next reading
    time.sleep(1)
