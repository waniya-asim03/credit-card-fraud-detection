import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt

# ===============================
# CONFIG
MODEL_FILE = "trained_random_forest_model.pkl"
DATA_FILE = "creditcard_small.csv"
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Credit_card_font_awesome.svg/1200px-Credit_card_font_awesome.svg.png"

# ===============================
# PAGE CONFIG (MUST BE FIRST)
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ===============================
# LOADERS

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Ensure required columns exist
    if "Time" not in df.columns:
        df.insert(0, "Time", 0)

    if "V1" not in df.columns:
        df.insert(1, "V1", 0)

    return df

# ===============================
# LOAD MODEL & DATA
model = load_model(MODEL_FILE)
data = load_data(DATA_FILE)

# 🔴 CRITICAL FIX: Align CSV with model features
MODEL_FEATURES = model.feature_names_in_
data = data.reindex(
    columns=list(MODEL_FEATURES) + ["Class"],
    fill_value=0
)

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# ===============================
# HELPERS (NO CACHING HERE ❌)

def compute_probs(model, X):
    return model.predict_proba(X)[:, 1]

def pick_transaction(df, random_pick, index):
    if random_pick:
        index = random.randint(0, len(df) - 1)
    return df.iloc[index], index

def highlight_features(row, numeric_cols, full_data):
    styles = []
    for col in row.index:
        if col in numeric_cols:
            if row[col] > full_data[col].mean() + 3 * full_data[col].std():
                styles.append("color:red; font-weight:bold")
            else:
                styles.append("")
        else:
            styles.append("")
    return styles

def get_risk(prob):
    if prob >= 0.8:
        return "High Risk ⚠️", "#1E90FF"
    elif prob >= 0.5:
        return "Medium Risk ⚠️", "#75C3F4"
    else:
        return "Low Risk ✅", "#82D9E5"

# ===============================
# HEADER
st.markdown(
    f"""
    <div style='background-color:#1E90FF;padding:15px;border-radius:10px;display:flex;align-items:center;'>
        <img src="{LOGO_URL}" width="50" style="margin-right:15px;">
        <div>
            <h1 style='color:white;margin:0;'>💳 Credit Card Fraud Detection</h1>
            <p style='color:white;margin:0;'>Interactive ML Fraud Detection App</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# SIDEBAR
row_index = 0

with st.sidebar:
    st.header("Options & Dataset Summary")
    use_random = st.checkbox("Pick random transaction", value=True)

    if not use_random:
        row_index = st.number_input(
            f"Row index (0 to {len(data)-1})",
            min_value=0,
            max_value=len(data)-1,
            value=0,
            step=1
        )

    st.subheader("Dataset Stats")
    st.write(f"Total Transactions: {len(data):,}")
    st.write(f"Frauds: {int(data['Class'].sum()):,}")
    st.write(f"Non-Frauds: {len(data) - int(data['Class'].sum()):,}")
    st.write(f"Fraud %: {data['Class'].mean() * 100:.2f}%")

    st.subheader("🚨 Top 5 High-Risk Transactions")
    X = data.drop(columns=["Class"])
    probs = compute_probs(model, X)

    temp = data.copy()
    temp["Fraud_Probability"] = probs

    st.dataframe(
        temp.sort_values("Fraud_Probability", ascending=False)
        .head(5)[["Fraud_Probability"]]
    )

# ===============================
# MAIN
transaction, row_index = pick_transaction(data, use_random, row_index)

st.subheader(f"Selected Transaction (Row {row_index})")
st.dataframe(
    transaction.to_frame().T.style.apply(
        highlight_features,
        axis=1,
        numeric_cols=numeric_cols,
        full_data=data
    )
)

# ===============================
# PREDICTION
if st.button("Predict"):
    features = transaction.drop("Class").values.reshape(1, -1)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    risk, color = get_risk(prob)

    st.markdown(
        f"<h3 style='color:{color}'>Prediction: {risk} ({prob:.2%})</h3>",
        unsafe_allow_html=True
    )

    st.write(f"Predicted Class: {prediction}")

# ===============================
# CHART
st.subheader("Fraud vs Non-Fraud Distribution")
fig, ax = plt.subplots()
data["Class"].value_counts().plot.pie(
    autopct="%1.1f%%",
    labels=["Not Fraud", "Fraud"],
    colors=["#87CEEB", "#1E90FF"],
    ax=ax
)
ax.set_ylabel("")
st.pyplot(fig)

# ===============================
# DOWNLOAD
st.subheader("📄 Download Transaction Report")
csv = transaction.to_frame().T.to_csv(index=False)
st.download_button(
    "Download CSV",
    csv,
    file_name=f"transaction_{row_index}.csv",
    mime="text/csv"
)
