import joblib
import pandas as pd

# Load trained model
model = joblib.load("machine_failure_model.pkl")

# New machine readings
data = pd.DataFrame([{
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
}])

# Get failure probability
probability = model.predict_proba(data)[0][1]
percentage = probability * 100

# Determine risk level
if percentage < 30:
    status = "🟢 NORMAL"
elif percentage < 70:
    status = "🟡 WARNING"
else:
    status = "🔴 CRITICAL"

# Display result
print(f"Machine Failure Probability: {percentage:.2f}%")
print(f"Risk Level: {status}")

# Recommendation
if percentage < 30:
    print("Recommendation: Continue normal operation.")
elif percentage < 70:
    print("Recommendation: Monitor the machine closely.")
else:
    print("Recommendation: Inspect the machine and consider maintenance.")
