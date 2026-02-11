import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import shap
from streamlit_shap import st_shap  # Updated import
import plotly.graph_objects as go

# ===============================
# CONFIG & LOADERS
BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "trained_random_forest_model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "creditcard_small.csv")
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Credit_card_font_awesome.svg/1200px-Credit_card_font_awesome.svg.png"

@st.cache_resource
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def get_global_stats(_model, _data):
    """Pre-calculate probabilities for the whole set once for performance."""
    X = _data.reindex(columns=_model.feature_names_in_, fill_value=0)
    probs = _model.predict_proba(X)[:, 1]
    temp = _data.copy()
    temp["Fraud_Probability"] = probs
    return temp, probs

# ===============================
# INITIALIZE
st.set_page_config(page_title="FraudShield AI", page_icon="💳", layout="wide")
model = load_model(MODEL_FILE)
raw_data = load_data(DATA_FILE)

if raw_data.empty or model is None:
    st.error("❌ Missing Model or Data files. Please check your directory.")
    st.stop()

# Align data
MODEL_FEATURES = list(model.feature_names_in_)
data, all_probs = get_global_stats(model, raw_data)

# ===============================
# SIDEBAR
with st.sidebar:
    st.title("🛡️ FraudShield Ops")
    threshold = st.slider("Risk Threshold Sensitivity", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.subheader("System Stats")
    st.metric("Total Records", f"{len(data):,}")
    st.metric("Avg Dataset Risk", f"{all_probs.mean():.2%}")
    
    st.subheader("High Priority Alerts 🚨")
    alerts = data.sort_values("Fraud_Probability", ascending=False).head(5)
    st.dataframe(alerts[["Fraud_Probability"]].style.format("{:.2%}"), use_container_width=True)

# ===============================
# MAIN UI
st.markdown(f"""
    <div style='background-color:#1E90FF;padding:15px;border-radius:10px;color:white;display:flex;align-items:center;'>
        <img src="{LOGO_URL}" width="40" style="margin-right:15px;">
        <h1 style='margin:0;'>Fraud Detection Command Center</h1>
    </div>""", unsafe_allow_html=True)

st.write("") # Spacer

# Selection Logic
use_random = st.checkbox("Automated Random Sampling", value=True)
if not use_random:
    row_idx = st.number_input("Manual Row ID", 0, len(data)-1, value=0)
else:
    # We use session state to keep the random index stable until 'New Sample' is clicked
    if 'rand_idx' not in st.session_state:
        st.session_state.rand_idx = random.randint(0, len(data)-1)
    if st.button("🔄 Get New Sample"):
        st.session_state.rand_idx = random.randint(0, len(data)-1)
    row_idx = st.session_state.rand_idx

transaction = data.iloc[row_idx]

# --- PREDICTION & RADAR SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Metadata")
    st.dataframe(transaction.to_frame().T[MODEL_FEATURES].style.format(precision=3), use_container_width=True)
    
    prob = transaction["Fraud_Probability"]
    
    # Dynamic Risk Assessment based on Slider
    if prob >= threshold:
        risk_label, color = "CRITICAL RISK ⚠️", "#FF4C4C"
    elif prob >= (threshold / 2):
        risk_label, color = "ELEVATED RISK ⚠️", "#FFA500"
    else:
        risk_label, color = "LOW RISK ✅", "#32CD32"

    st.markdown(f"""
        <div style='border:2px solid {color}; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:{color}; margin:0;'>{risk_label}</h2>
            <h3 style='margin:0;'>{prob:.2%} Fraud Match</h3>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Feature Variance Profile")
    # Radar chart comparing current to mean for key features
    # V17, V14, V12, V10, V16 are often high-impact in fraud datasets
    top_v = ["V17", "V14", "V12", "V10", "V11"] 
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=transaction[top_v].values, theta=top_v, fill='toself', name='Current Row', line_color='#1E90FF'))
    fig.add_trace(go.Scatterpolar(r=data[top_v].mean().values, theta=top_v, name='Dataset Avg', line_color='#FF4C4C'))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-4, 4])),
        showlegend=True, height=350, margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- SHAP EXPLAINABILITY ---
st.markdown("---")
st.subheader("🔍 AI Logic Decomposition (SHAP Waterfall)")
st.write("This chart explains exactly how each feature contributed to the final probability score.")

# SHAP calculation
explainer = shap.TreeExplainer(model)
# Note: transaction[MODEL_FEATURES] ensures we don't pass 'Class' or 'Fraud_Probability'
shap_values = explainer(transaction[MODEL_FEATURES].to_frame().T)

# For Binary Classification in RF, SHAP usually returns a list [Class 0, Class 1] 
# or a single array. We want to explain the 'Fraud' probability (Class 1).
if isinstance(shap_values, list):
    st_shap(shap.plots.waterfall(shap_values[1][0]), height=400)
else:
    # In newer SHAP versions, it might be a multi-output object
    st_shap(shap.plots.waterfall(shap_values[0]), height=400)



# --- DATA DISTRIBUTION ---
with st.expander("📊 Dataset Distribution Overview"):
    fig2, ax2 = plt.subplots(figsize=(4,4))
    data['Class'].value_counts().plot.pie(
        labels=['Safe', 'Fraud'], autopct='%1.1f%%', colors=['#87CEEB','#1E90FF'], ax=ax2
    )
    ax2.set_ylabel('')
    st.pyplot(fig2)
