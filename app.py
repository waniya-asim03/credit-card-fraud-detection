import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import os
import shap
from streamlit_shap import st_shap 
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ===============================
# CONFIG & LOADERS
BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "trained_random_forest_model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "creditcard_small.csv")
# This is the icon you asked about!
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Credit_card_font_awesome.svg/1200px-Credit_card_font_awesome.svg.png"

@st.cache_resource
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def get_processed_data(_model, _data_path):
    if not os.path.exists(_data_path):
        return pd.DataFrame(), None
    df = pd.read_csv(_data_path)
    model_cols = list(_model.feature_names_in_)
    # Defensive fix for Time/V1 missing columns
    df_aligned = df.reindex(columns=model_cols, fill_value=0)
    probs = _model.predict_proba(df_aligned)[:, 1]
    df_aligned["Fraud_Probability"] = probs
    if "Class" in df.columns:
        df_aligned["Class"] = df["Class"].values
    return df_aligned, probs

# ===============================
# INITIALIZE
st.set_page_config(page_title="FraudShield AI", page_icon="💳", layout="wide")
model = load_model(MODEL_FILE)
data, all_probs = get_processed_data(model, DATA_FILE)

if data.empty or model is None:
    st.error("❌ Model or CSV not found. Ensure both are in your folder.")
    st.stop()

MODEL_FEATURES = list(model.feature_names_in_)

# ===============================
# SIDEBAR
with st.sidebar:
    st.title("🛡️ FraudShield Ops")
    threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.subheader("System Stats")
    st.metric("Total Records", f"{len(data):,}")
    st.metric("Fraud Cases", f"{int(data['Class'].sum())}")
    
    st.subheader("High Risk Queue 🚨")
    alerts = data.sort_values("Fraud_Probability", ascending=False).head(5)
    st.table(alerts[["Fraud_Probability"]].style.format("{:.1%}"))

# ===============================
# MAIN UI HEADER
st.markdown(f"""
    <div style='background-color:#1E90FF;padding:15px;border-radius:10px;color:white;display:flex;align-items:center;'>
        <img src="{LOGO_URL}" width="40" style="margin-right:15px;">
        <h1 style='margin:0;'>Fraud Detection Command Center</h1>
    </div>""", unsafe_allow_html=True)

st.write("") 

# --- SELECTION LOGIC ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    mode = st.radio("Selection Mode", ["Random", "Manual ID"], horizontal=True)
with c2:
    if mode == "Random" and st.button("🔄 Pick Any Row"):
        st.session_state.idx = random.randint(0, len(data)-1)
with c3:
    # THIS FIXES YOUR "ALWAYS LOW" PROBLEM:
    if mode == "Random" and st.button("🧪 Test a Fraud Case"):
        fraud_indices = data[data['Class'] == 1].index.tolist()
        if fraud_indices:
            st.session_state.idx = random.choice(fraud_indices)
        else:
            st.warning("No Fraud cases found in CSV.")

if mode == "Manual ID":
    st.session_state.idx = st.number_input("Enter Row ID", 0, len(data)-1, value=st.session_state.idx)

transaction = data.iloc[st.session_state.idx]

# --- PREDICTION & RADAR ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Transaction Profile (ID: {st.session_state.idx})")
    st.dataframe(transaction.to_frame().T[MODEL_FEATURES].style.format(precision=3), use_container_width=True)
    
    # Run prediction check
    prob = transaction["Fraud_Probability"]
    color = "#FF4C4C" if prob >= threshold else ("#FFA500" if prob >= threshold/2 else "#32CD32")
    risk = "CRITICAL RISK 🚨" if prob >= threshold else ("ELEVATED RISK ⚠️" if prob >= threshold/2 else "SAFE / LOW RISK ✅")

    st.markdown(f"""
        <div style='background-color:{color}; padding:25px; border-radius:15px; text-align:center; color:white; border: 4px solid rgba(255,255,255,0.3);'>
            <h1 style='margin:0; font-size: 40px;'>{risk}</h1>
            <h2 style='margin:0;'>{prob:.1%} Fraud Match</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Visual Feature Signature")
    # Radar chart comparing current to dataset mean
    radar_feats = ["V17", "V14", "V12", "V10", "V11"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=transaction[radar_feats].values, theta=radar_feats, fill='toself', name='Target Row', line_color='white'))
    fig.add_trace(go.Scatterpolar(r=data[radar_feats].mean().values, theta=radar_feats, name='Average Case', line_color='yellow'))
    fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False, range=[-7, 7])), height=350, margin=dict(l=40, r=40, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --- SHAP SECTION (FIXED HEIGHT) ---
st.markdown("---")
st.subheader("🔍 Logic Breakdown: Why the AI assigned this score")
st.write("Red features increase risk; Blue features decrease risk.")

explainer = shap.TreeExplainer(model)
shap_values = explainer(transaction[MODEL_FEATURES].to_frame().T)

# Handle different SHAP output formats (Classes vs Probs)
if len(shap_values.shape) == 3:
    target_shap = shap_values[0, :, 1]
else:
    target_shap = shap_values[0]

# Increased height and limited features to prevent "half-picture" issue

st_shap(shap.plots.waterfall(target_shap, max_display=10), height=550)

# --- FOOTER ---
with st.expander("📊 Global Context"):
    st.write("Distribution of Fraud vs Legitimate transactions in your current dataset.")
    fig2, ax2 = plt.subplots(figsize=(4,4))
    data['Class'].value_counts().plot.pie(labels=['Safe', 'Fraud'], autopct='%1.1f%%', colors=['#87CEEB','#1E90FF'], ax=ax2)
    ax2.set_ylabel('')
    st.pyplot(fig2)
