# STEP 49: Build the Streamlit Interface
import streamlit as st
import joblib 
from pathlib import Path
import pandas as pd

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

st.set_page_config(
    page_title="Equipment Fault Detection",
    page_icon="⚙️",
    layout="wide"
)
st.title("Equipment Fault Detection")
st.caption("Machine Learning-Based Equipment Fault Classification")

col1, col2 = st.columns(2)
with col1:
    machine_type = st.selectbox("Product Type", ["L", "M", "H"])

    air_temp = st.number_input("Air Temperature (K)", value=298.1, min_value=0.0, step=0.1)

    process_temp = st.number_input("Process Temperature (K)", value=308.6, min_value=0.0, step=0.1)

with col2:
    rotational_speed = st.number_input("Rotational Speed (RPM)", value=1500, min_value=0, step=1)

    torque = st.number_input("Torque (Nm)", value=45.0, min_value=0.0, step=0.1)

    tool_wear = st.number_input("Tool Wear (min)", value=150, min_value=0, step=1)


st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset: AI4I 2020")
st.sidebar.info(
    "Educational prototype trained on synthetic data. "
    "Predictions are not validated for real industrial equipment."
)


input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear]
    })

st.dataframe(input_data)

model_path = Path("artifacts/my_random_forest.joblib")

# Load the saved file using joblib.load(). Store it in saved_bundle
saved_bundle = joblib.load(model_path)

# Retrieve the trained Random Forest. Store it in loaded_model
loaded_model = saved_bundle["model"]

# Retrieve the selected threshold. Store it in loaded_threshold
loaded_threshold = saved_bundle["threshold"]

loaded_ranges = saved_bundle["feature_ranges"]


#Generate Machine Failure Prediction
if st.button("Predict Machine Failure"):
    for feature, limits in loaded_ranges.items():
        user_value = input_data.iloc[0][feature]
        minimum = limits["min"]
        maximum = limits["max"]

        if user_value < minimum or user_value > maximum:
            st.warning(f"{feature}: {user_value} is outside the training "
                       f"range ({minimum} to {maximum})."
                       "The prediction range may be less reliable.")

    failure_probability = loaded_model.predict_proba(input_data)[0, 1]

    if failure_probability >= loaded_threshold:
        st.warning("Potential Machine Failure Detected")
    else:
        st.success("Normal Operation Predicted")

    st.metric(label="Model Failure Score",
              value=f"{failure_probability:.4%}")
    st.progress(failure_probability)
    st.caption(f"Classification Threshold: {loaded_threshold:.2%}")


numerical_features = FEATURES[1:]
