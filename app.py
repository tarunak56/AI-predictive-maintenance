import streamlit as st
import joblib
import pandas as pd
import time

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load("machine_failure_model.pkl")
df = pd.read_csv("ai4i2020.csv")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Machine Monitor",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🏭 AI Machine Failure Monitoring System")

st.write(
    "AI-powered predictive maintenance using machine operating data."
)

# ============================================================
# FAILURE TYPE DIAGNOSIS FUNCTION
# ============================================================

def diagnose_failure(air_temp, process_temp, speed, torque, tool_wear):

    # Temperature difference
    temperature_difference = process_temp - air_temp

    # --------------------------------------------------------
    # Power Failure
    # High torque + low rotational speed
    # --------------------------------------------------------

    if torque >= 60 and speed <= 1400:

        return (
            "⚡ Power Failure (PWF)",
            "Inspect motor load, torque and power transmission."
        )

    # --------------------------------------------------------
    # Overstrain Failure
    # High torque + high tool wear
    # --------------------------------------------------------

    elif torque >= 55 and tool_wear >= 180:

        return (
            "💥 Overstrain Failure (OSF)",
            "Inspect excessive mechanical load and tool condition."
        )

    # --------------------------------------------------------
    # Tool Wear Failure
    # High tool wear
    # --------------------------------------------------------

    elif tool_wear >= 200:

        return (
            "🔧 Tool Wear Failure (TWF)",
            "Inspect and replace the worn tool if necessary."
        )

    # --------------------------------------------------------
    # Heat Dissipation Failure
    # High temperature difference
    # --------------------------------------------------------

    elif temperature_difference >= 12:

        return (
            "🌡️ Heat Dissipation Failure (HDF)",
            "Inspect cooling system and heat dissipation."
        )

    # --------------------------------------------------------
    # Random Failure
    # --------------------------------------------------------

    else:

        return (
            "❓ Possible Random Failure (RNF)",
            "Perform a general machine inspection."
        )


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(
    "Select Monitoring Mode",
    [
        "🧑‍🔧 Manual Input",
        "🔄 Real-Time Simulation"
    ]
)

# ============================================================
# MANUAL INPUT MODE
# ============================================================

if mode == "🧑‍🔧 Manual Input":

    st.header("🧑‍🔧 Enter Machine Readings")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # MACHINE INPUTS
    # --------------------------------------------------------

    with col1:

        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=300.0
        )

        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=310.0
        )

        speed = st.number_input(
            "Rotational Speed [RPM]",
            min_value=500,
            max_value=3000,
            value=1500
        )

    with col2:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button("🔍 Predict Machine Risk"):

        data = pd.DataFrame([{

            "Air temperature [K]": air_temp,

            "Process temperature [K]": process_temp,

            "Rotational speed [rpm]": speed,

            "Torque [Nm]": torque,

            "Tool wear [min]": tool_wear

        }])

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        probability = model.predict_proba(data)[0][1]

        percentage = probability * 100

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if percentage < 30:

            status = "🟢 NORMAL"

            recommendation = (
                "Continue normal operation."
            )

        elif percentage < 70:

            status = "🟡 WARNING"

            recommendation = (
                "Monitor the machine closely."
            )

        else:

            status = "🔴 CRITICAL"

            recommendation = (
                "Inspect the machine and consider maintenance."
            )

        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("Prediction Result")

        st.metric(
            "Machine Failure Probability",
            f"{percentage:.2f}%"
        )

        st.write(
            f"### Risk Level: {status}"
        )

        # ====================================================
        # AUTOMATIC ALERT
        # ====================================================

        if percentage >= 70:

            st.error(
                "🚨 CRITICAL ALERT: High probability "
                "of machine failure!"
            )

        elif percentage >= 30:

            st.warning(
                "⚠️ WARNING ALERT: Machine risk is increasing."
            )

        else:

            st.success(
                "✅ SYSTEM NORMAL: Machine operating "
                "within normal risk level."
            )

        # ====================================================
        # FAILURE TYPE DIAGNOSIS
        # ====================================================

        if percentage >= 30:

            failure_type, action = diagnose_failure(
                air_temp,
                process_temp,
                speed,
                torque,
                tool_wear
            )

            st.subheader("🔧 Likely Failure Type")

            st.write(
                f"### {failure_type}"
            )

            st.info(
                f"**Recommended Action:** {action}"
            )

        # ====================================================
        # GENERAL RECOMMENDATION
        # ====================================================

        st.info(
            f"**Recommendation:** {recommendation}"
        )


# ============================================================
# REAL-TIME SIMULATION MODE
# ============================================================

else:

    st.header("🔄 Real-Time Machine Simulation")

    st.write(
        "Simulated sensor readings are streamed from the "
        "AI4I 2020 dataset."
    )

    # ========================================================
    # START SIMULATION
    # ========================================================

    if st.button("▶️ Start Simulation"):

        risk_history = []

        # ----------------------------------------------------
        # PLACEHOLDERS
        # ----------------------------------------------------

        reading_placeholder = st.empty()

        metrics_placeholder = st.empty()

        probability_placeholder = st.empty()

        status_placeholder = st.empty()

        alert_placeholder = st.empty()

        diagnosis_placeholder = st.empty()

        action_placeholder = st.empty()

        chart_placeholder = st.empty()

        # ====================================================
        # STREAM DATA
        # ====================================================

        for index, row in df.iterrows():

            # ------------------------------------------------
            # CREATE MODEL INPUT
            # ------------------------------------------------

            data = pd.DataFrame([{

                "Air temperature [K]":
                    row["Air temperature [K]"],

                "Process temperature [K]":
                    row["Process temperature [K]"],

                "Rotational speed [rpm]":
                    row["Rotational speed [rpm]"],

                "Torque [Nm]":
                    row["Torque [Nm]"],

                "Tool wear [min]":
                    row["Tool wear [min]"]

            }])

            # ------------------------------------------------
            # AI PREDICTION
            # ------------------------------------------------

            probability = model.predict_proba(data)[0][1]

            percentage = probability * 100

            # ------------------------------------------------
            # RISK CLASSIFICATION
            # ------------------------------------------------

            if percentage < 30:

                status = "🟢 NORMAL"

                recommendation = (
                    "Continue normal operation."
                )

            elif percentage < 70:

                status = "🟡 WARNING"

                recommendation = (
                    "Monitor the machine closely."
                )

            else:

                status = "🔴 CRITICAL"

                recommendation = (
                    "Inspect the machine and consider maintenance."
                )

            # ------------------------------------------------
            # STORE RISK
            # ------------------------------------------------

            risk_history.append(percentage)

            # =================================================
            # CURRENT READING
            # =================================================

            reading_placeholder.subheader(
                f"📡 Current Machine Reading: #{index + 1}"
            )

            # =================================================
            # SENSOR VALUES
            # =================================================

            with metrics_placeholder.container():

                col1, col2, col3, col4, col5 = st.columns(5)

                col1.metric(
                    "Air Temp",
                    f"{row['Air temperature [K]']:.1f} K"
                )

                col2.metric(
                    "Process Temp",
                    f"{row['Process temperature [K]']:.1f} K"
                )

                col3.metric(
                    "Speed",
                    f"{row['Rotational speed [rpm]']:.0f} RPM"
                )

                col4.metric(
                    "Torque",
                    f"{row['Torque [Nm]']:.1f} Nm"
                )

                col5.metric(
                    "Tool Wear",
                    f"{row['Tool wear [min]']:.0f} min"
                )

            # =================================================
            # FAILURE PROBABILITY
            # =================================================

            probability_placeholder.metric(
                "⚠️ Failure Probability",
                f"{percentage:.2f}%"
            )

            # =================================================
            # STATUS
            # =================================================

            status_placeholder.write(
                f"## Status: {status}"
            )

            # =================================================
            # AUTOMATIC ALERT
            # =================================================

            if percentage >= 70:

                alert_placeholder.error(
                    "🚨 CRITICAL ALERT: High probability "
                    "of machine failure detected!"
                )

            elif percentage >= 30:

                alert_placeholder.warning(
                    "⚠️ WARNING ALERT: Machine risk "
                    "is increasing. Monitor closely."
                )

            else:

                alert_placeholder.success(
                    "✅ SYSTEM NORMAL: Machine operating "
                    "within normal risk level."
                )

            # =================================================
            # FAILURE TYPE DIAGNOSIS
            # =================================================

            if percentage >= 30:

                failure_type, action = diagnose_failure(

                    row["Air temperature [K]"],

                    row["Process temperature [K]"],

                    row["Rotational speed [rpm]"],

                    row["Torque [Nm]"],

                    row["Tool wear [min]"]

                )

                diagnosis_placeholder.write(
                    f"### 🔧 Likely Failure Type: {failure_type}"
                )

                action_placeholder.info(
                    f"**Recommended Action:** {action}"
                )

            else:

                diagnosis_placeholder.write(
                    "### 🔧 Failure Diagnosis: None"
                )

                action_placeholder.info(
                    "No immediate maintenance action required."
                )

            # =================================================
            # GENERAL RECOMMENDATION
            # =================================================

            st.write(
                f"**Recommendation:** {recommendation}"
            )

            # =================================================
            # LIVE GRAPH
            # =================================================

            chart_placeholder.line_chart(

                pd.DataFrame(
                    {
                        "Failure Risk (%)":
                            risk_history
                    }
                )

            )

            # =================================================
            # WAIT
            # =================================================

            time.sleep(1)
