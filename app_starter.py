"""Local Streamlit app: a simple equipment-failure classification demo."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from train import FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'artifacts' / 'fault_model.joblib'
METRICS_PATH = ROOT / 'reports' / 'metrics.csv'

st.set_page_config(page_title='Equipment Fault Detection', page_icon='⚙️', layout='centered')
st.title('⚙️ Equipment Fault Detection')
st.caption('Educational machine-failure classifier • AI4I 2020 synthetic dataset')
st.markdown('Enter operating measurements to estimate the probability that this observation is labeled a machine failure.')

if not MODEL_PATH.exists():
    st.error('Model not found. First run `python download_dataset.py` and `python train.py` in a terminal.')
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)  # Only load models generated locally and trusted.

model = load_model()
threshold = st.slider('Failure alert threshold', min_value=0.05, max_value=0.95, value=0.50, step=0.05)

with st.form('input_form'):
    st.subheader('Current operating measurements')
    machine_type = st.selectbox('Product type', ['L', 'M', 'H'], help='L=low, M=medium, H=high product quality variant')
    air_temp = st.number_input('Air temperature (K)', min_value=260.0, max_value=360.0, value=300.0, step=0.1)
    process_temp = st.number_input('Process temperature (K)', min_value=260.0, max_value=380.0, value=310.0, step=0.1)
    rpm = st.number_input('Rotational speed (rpm)', min_value=0, max_value=10000, value=1500, step=10)
    torque = st.number_input('Torque (Nm)', min_value=0.0, max_value=200.0, value=40.0, step=0.5)
    wear = st.number_input('Tool wear (minutes)', min_value=0, max_value=1000, value=100, step=5)
    predict = st.form_submit_button('Predict machine failure')

if predict:
    record = pd.DataFrame([{
        'Type': machine_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': wear,
    }], columns=FEATURES)
    prob = float(model.predict_proba(record)[0, list(model.classes_).index(1)])
    st.metric('Model failure probability (uncalibrated)', f'{prob:.1%}')
    st.progress(prob)
    if prob >= threshold:
        st.warning('Classification: POSSIBLE MACHINE FAILURE')
    else:
        st.success('Classification: NORMAL OPERATION')
    st.caption(f'Threshold: {threshold:.0%}. Model probabilities have not been calibrated on real equipment; this is not a guarantee.')

st.divider()
st.subheader('Batch predictions (optional)')
st.write('Upload a CSV with these six columns: ' + ', '.join(f'`{f}`' for f in FEATURES))
uploaded = st.file_uploader('Upload operating measurements CSV', type=['csv'])
if uploaded:
    try:
        incoming = pd.read_csv(uploaded)
        missing = [col for col in FEATURES if col not in incoming.columns]
        if missing:
            st.error('Missing required columns: ' + ', '.join(missing))
        elif incoming.empty:
            st.error('The uploaded CSV contains no rows.')
        elif not incoming['Type'].astype(str).isin(['L', 'M', 'H']).all():
            st.error('Type must contain only L, M, H.')
        else:
            clean = incoming[FEATURES].copy()
            for col in FEATURES[1:]:
                clean[col] = pd.to_numeric(clean[col], errors='raise')
            if clean.isna().any().any():
                raise ValueError('Missing values are not supported in this demo upload')
            proba = model.predict_proba(clean)[:, list(model.classes_).index(1)]
            output = incoming.copy()
            output['failure_probability'] = proba.round(4)
            output['predicted_failure'] = (proba >= threshold).astype(int)
            st.dataframe(output.head(100), use_container_width=True)
            st.download_button('Download predictions CSV', data=output.to_csv(index=False).encode('utf-8'), file_name='machine_predictions.csv', mime='text/csv')
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f'Invalid CSV: {exc}')

if METRICS_PATH.exists():
    st.divider()
    st.subheader('Evaluation on held-out test data')
    results = pd.read_csv(METRICS_PATH)
    st.dataframe(results[['model', 'failure_precision', 'failure_recall', 'failure_f1', 'accuracy']], hide_index=True, use_container_width=True)

st.info('Limitations: the UCI data are synthetic. The label describes failure for an observation, not time until a future failure. This prototype must not control real equipment or replace safety monitoring.')
st.markdown('Dataset credit: [AI4I 2020, UCI Machine Learning Repository](https://doi.org/10.24432/C5HS5C), CC BY 4.0.')
