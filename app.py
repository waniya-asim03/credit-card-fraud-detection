import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import shap
from streamlit_shap import st_shap 
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
def get_processed_data(_model, _data):
    """
    Ensures data has all columns the model expects (fixes the KeyError).
    """
    model_cols = list(_model.feature_names_in_)
    # Reindex fills missing columns (Time, V1) with 0
    df_aligned = _data.reindex(columns=model_cols, fill_value=0)
    
    # Calculate probabilities for the entire set once
    probs = _model.predict_proba(df_aligned)[:, 1]
    df_aligned["Fraud_Probability"] = probs
    
    if "Class" in _data.columns:
        df_aligned["Class"] = _data["Class"].values
    else:
        df_aligned["Class"] = 0
        
    return df_aligned, probs

# ===============================
# INITIALIZE
st.set_page_config(page_title="FraudShield AI", page_icon="💳", layout="wide")
model = load_model(MODEL_FILE)
raw_data = load_data(DATA_FILE)

if raw_data.empty or model is None:
    st.error("❌ Missing Model or Data files.")
    st.stop()

data, all_probs = get_processed_data(model, raw_data)
MODEL_FEATURES = list(model.feature_names_in_)

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

st.write("") 

# Stable Random Selection logic
if 'rand_idx' not in st.session_state:
    st.session_state.rand_idx = random.randint(0, len(data)-1)

col_check, col_btn = st.columns([2, 1])
with col_check:
    use_random = st.checkbox("Automated Random Sampling", value=True)
with col_btn:
    if use_random and st.button("🔄 New Sample"):
        st.session_state.rand_idx = random.randint(0, len(data)-1)

row_idx = st.session_state.rand_idx if use_random else st.number_input("Manual Row ID", 0, len(data)-1, value=0)
transaction = data.iloc[row_idx]

# --- PREDICTION & RADAR SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Metadata")
    st.dataframe(transaction.to_frame().T[MODEL_FEATURES].style.format(precision=3), use_container_width=True)
    
    # PREDICT BUTTON
    if st.button("Run Forensic Analysis", use_container_width=True):
        prob = transaction["Fraud_Probability"]
        color = "#FF4C4C" if prob >= threshold else ("#FFA500" if prob >= threshold/2 else "#32CD32")
        risk_label = "CRITICAL RISK ⚠️" if prob >= threshold else ("ELEVATED RISK ⚠️" if prob >= threshold/2 else "LOW RISK ✅")

        st.markdown(f"""
            <div style='border:2px solid {color}; padding:20px; border-radius:10px; text-align:center;'>
                <h2 style='color:{color}; margin:0;'>{risk_label}</h2>
                <h3 style='margin:0;'>{prob:.2%} Fraud Match</h3>
            </div>
        """, unsafe_allow_html=True)

        # --- SHAP EXPLAINABILITY (Inside Button) ---
        st.markdown("---")
        st.subheader("🔍 AI Logic Decomposition (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_input = transaction[MODEL_FEATURES].to_frame().T
        shap_values = explainer(shap_input)
        
        # Handling the shape of SHAP values for Random Forest
        if len(shap_values.shape) == 3: 
            st_shap(shap.plots.waterfall(shap_values[0, :, 1]), height=400)
        else:
            st_shap(shap.plots.waterfall(shap_values[0]), height=400)

with col2:
    st.subheader("Feature Variance Profile")
    # Using key V-features for the radar plot
    radar_feats = ["V17", "V14", "V12", "V10", "V11"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=transaction[radar_feats].values, theta=radar_feats, fill='toself', name='Current', line_color='#1E90FF'))
    fig.add_trace(go.Scatterpolar(r=data[radar_feats].mean().values, theta=radar_feats, name='Dataset Avg', line_color='#FF4C4C'))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-5, 5])),
        showlegend=True, height=350, margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Waterfall Plot
    

# --- DATA DISTRIBUTION ---
with st.expander("📊 Global Dataset Context"):
    fig2, ax2 = plt.subplots(figsize=(4,4))
    data['Class'].value_counts().plot.pie(
        labels=['Safe', 'Fraud'], autopct='%1.1f%%', colors=['#87CEEB','#1E90FF'], ax=ax2
    )
    ax2.set_ylabel('')
    st.pyplot(fig2)
