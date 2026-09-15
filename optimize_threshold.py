import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("ai4i2020.csv")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

X = df[features]
y = df[target]

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# 3. TRAIN GRADIENT BOOSTING MODEL
# ============================================================

print("=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

print("\nTraining Gradient Boosting model...")

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# 4. GET FAILURE PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]

# ============================================================
# 5. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

results = []

print("\n")
print("=" * 60)
print("TESTING THRESHOLDS")
print("=" * 60)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)

# ============================================================
# 7. FIND BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n")
print("=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_row['Threshold']:.2f}"
)

print(
    f"Accuracy  : {best_row['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision : {best_row['Precision'] * 100:.2f}%"
)

print(
    f"Recall    : {best_row['Recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : {best_row['F1'] * 100:.2f}%"
)

# ============================================================
# 8. FIND BEST RECALL THRESHOLD
# ============================================================

best_recall_row = results_df.loc[
    results_df["Recall"].idxmax()
]

print("\n")
print("=" * 60)
print("HIGHEST RECALL THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_recall_row['Threshold']:.2f}"
)

print(
    f"Accuracy  : {best_recall_row['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision : {best_recall_row['Precision'] * 100:.2f}%"
)

print(
    f"Recall    : {best_recall_row['Recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : {best_recall_row['F1'] * 100:.2f}%"
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "threshold_results.csv",
    index=False
)

# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "optimized_machine_failure_model.pkl"
)

print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print("optimized_machine_failure_model.pkl")
print("threshold_results.csv")

print("\nThreshold optimization completed! 🚀")
