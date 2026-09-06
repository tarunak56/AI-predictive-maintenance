# AI-predictive-maintenance
AI-powered real-time machine failure detection and predictive maintenance system using simulated sensor data.
# AI Predictive Maintenance

AI-powered real-time machine failure detection and predictive maintenance system using simulated sensor data.

## 🚀 Overview

Machine failures can cause production delays, maintenance costs, and unexpected downtime. This project uses Machine Learning to analyze machine operating conditions and detect the possibility of failure before it occurs.

The system uses the AI4I 2020 Predictive Maintenance Dataset and simulates real-time sensor data streaming to demonstrate how an AI-based predictive maintenance system can work in an industrial environment.

## 🎯 Problem Statement

Traditional maintenance approaches often depend on fixed maintenance schedules or detecting failures after they happen.

Our system aims to:

- Monitor machine operating conditions
- Detect abnormal operating patterns
- Predict potential machine failures
- Generate a machine health/risk level
- Provide actionable maintenance recommendations
- Display the results through a real-time dashboard

## 🧠 Machine Learning

The project uses a Random Forest classification model.

### Input Features

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

### Target

`Machine failure`

- `0` → No failure
- `1` → Machine failure

## 📊 Dataset

We use the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains 10,000 machine observations with operating-condition and failure-related information.

Dataset:
https://archive.ics.uci.edu/dataset/601/ai4i

## ⚙️ System Architecture

Public Dataset
↓
Data Preprocessing
↓
Machine Learning Model
↓
Simulated Real-Time Data Stream
↓
Failure Prediction
↓
Risk Assessment
↓
Alert & Recommendation
↓
Real-Time Dashboard

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Git & GitHub
