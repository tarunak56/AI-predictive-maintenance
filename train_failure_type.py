import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("ai4i2020.csv")


# ==========================================
# 2. SELECT SENSOR FEATURES
# ==========================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ==========================================
# 3. FAILURE TYPES
# ==========================================

failure_types = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


# ==========================================
# 4. KEEP ONLY MACHINE FAILURE ROWS
# ==========================================

failure_df = df[df["Machine failure"] == 1].copy()


# ==========================================
# 5. REMOVE MULTIPLE-FAILURE ROWS
# ==========================================

failure_df["failure_count"] = failure_df[failure_types].sum(axis=1)

failure_df = failure_df[
    failure_df["failure_count"] == 1
].copy()


# ==========================================
# 6. IDENTIFY FAILURE TYPE
# ==========================================

def get_failure_type(row):

    for failure_type in failure_types:

        if row[failure_type] == 1:
            return failure_type

    return None


failure_df["Failure Type"] = failure_df.apply(
    get_failure_type,
    axis=1
)


# Remove rows without a failure type
failure_df = failure_df.dropna(
    subset=["Failure Type"]
)


# ==========================================
# 7. DISPLAY DATASET INFORMATION
# ==========================================

print("====================================")
print("FAILURE TYPE AI MODEL")
print("====================================")

print(
    "\nTotal training samples:",
    len(failure_df)
)

print("\nFailure type distribution:")

print(
    failure_df["Failure Type"].value_counts()
)


# ==========================================
# 8. CREATE X AND Y
# ==========================================

X = failure_df[features]

y = failure_df["Failure Type"]


# ==========================================
# 9. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 10. CREATE RANDOM FOREST
# ==========================================

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced"
)


# ==========================================
# 11. TRAIN AI
# ==========================================

print("\nTraining Failure-Type AI...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# 12. TEST AI
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 13. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

print("\nFailure Type Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 15. FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")

for feature, importance in zip(
    features,
    model.feature_importances_
):

    print(
        f"{feature}: {importance:.4f}"
    )


# ==========================================
# 16. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "failure_type_model.pkl"
)


print("\n====================================")
print("✅ FAILURE TYPE AI MODEL SAVED")
print("====================================")

print(
    "\nFile created:"
)

print(
    "failure_type_model.pkl"
)
