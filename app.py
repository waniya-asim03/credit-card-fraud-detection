import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ===============================
# CONFIG - Use absolute paths
BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "trained_random_forest_model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "creditcard_small.csv")
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Credit_card_font_awesome.svg/1200px-Credit_card_font_awesome.svg.png"

# ===============================
# PAGE CONFIG
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ===============================
# STYLING
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 10px;
        height: 40px;
        width: 100%;
    }
    .stDownloadButton>button {
        background-color: #75C3F4;
        color: white;
        border-radius: 10px;
        height: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOADERS

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    return joblib.load(path)

@st.cache_data
def load_data(path):
    """
    Load CSV data with safety checks.
    If file not found, allow user to upload it.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        st.warning(f"⚠️ CSV file not found: {path}")
        uploaded_file = st.file_uploader(
            "Upload your credit card CSV (max 22 MB)", type="csv"
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()
            return pd.DataFrame()

    # Ensure minimal required columns
    for col in ["Time", "V1"]:
        if col not in df.columns:
            df.insert(0 if col=="Time" else 1, col, 0)
    return df

# ===============================
# LOAD MODEL & DATA
model = load_model(MODEL_FILE)
data = load_data(DATA_FILE)

if data.empty or model is None:
    st.stop()  # Stop app if files not loaded

# Align CSV with model features
MODEL_FEATURES = model.feature_names_in_
data = data.reindex(columns=list(MODEL_FEATURES) + ["Class"], fill_value=0)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# ===============================
# HELPERS
def compute_probs(model, X):
    return model.predict_proba(X)[:, 1]

def pick_transaction(df, random_pick, index):
    if random_pick:
        index = random.randint(0, len(df) - 1)
    return df.iloc[index], index

def highlight_features(row, numeric_cols, full_data, top_features=[]):
    styles = []
    for col in row.index:
        if col in numeric_cols:
            if col in top_features:
                styles.append("color:red; font-weight:bold; background-color:#FFF0F0")
            elif row[col] > full_data[col].mean() + 3 * full_data[col].std():
                styles.append("color:red; font-weight:bold")
            else:
                styles.append("")
        else:
            styles.append("")
    return styles

def get_risk(prob):
    if prob >= 0.8:
        return "High Risk ⚠️", "#FF4C4C"
    elif prob >= 0.5:
        return "Medium Risk ⚠️", "#FFA500"
    else:
        return "Low Risk ✅", "#32CD32"

def top_unusual_features(row, numeric_cols, full_data, top_n=3):
    deviations = {}
    for col in numeric_cols:
        mean = full_data[col].mean()
        std = full_data[col].std()
        if std == 0:
            continue
        z = abs((row[col] - mean)/std)
        deviations[col] = z
    top_features = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [f[0] for f in top_features], pd.DataFrame(top_features, columns=["Feature", "Deviation (z-score)"])

# ===============================
# HEADER
st.markdown(
    f"""
    <div style='background-color:#1E90FF;padding:20px;border-radius:15px;display:flex;align-items:center;'>
        <img src="{LOGO_URL}" width="60" style="margin-right:20px;">
        <div>
            <h1 style='color:white;margin:0;'>💳 Credit Card Fraud Detection</h1>
            <p style='color:white;margin:0;font-size:16px;'>Interactive ML Fraud Detection App</p>
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
            f"Select Row Index (0 to {len(data)-1})",
            min_value=0,
            max_value=len(data)-1,
            value=0,
            step=1
        )
    
    st.subheader("Dataset Stats 📊")
    total_frauds = int(data['Class'].sum())
    st.write(f"Total Transactions: {len(data):,}")
    st.write(f"Frauds: {total_frauds:,}")
    st.write(f"Non-Frauds: {len(data) - total_frauds:,}")
    st.write(f"Fraud %: {data['Class'].mean() * 100:.2f}%")

    st.subheader("Top 5 High-Risk Transactions 🚨")
    X = data.drop(columns=["Class"])
    probs = compute_probs(model, X)
    temp = data.copy()
    temp["Fraud_Probability"] = probs
    st.dataframe(
        temp.sort_values("Fraud_Probability", ascending=False)
        .head(5)[["Fraud_Probability"]],
        use_container_width=True
    )

# ===============================
# MAIN TRANSACTION
transaction, row_index = pick_transaction(data, use_random, row_index)

# Get top unusual features
top_features, top_features_df = top_unusual_features(transaction, numeric_cols, data)

st.subheader(f"Selected Transaction (Row {row_index})")
st.dataframe(
    transaction.to_frame().T.style.apply(
        highlight_features,
        axis=1,
        numeric_cols=numeric_cols,
        full_data=data,
        top_features=top_features
    )
)

# ===============================
# PREDICTION
if st.button("Predict Transaction Risk"):
    features = transaction.drop("Class").values.reshape(1, -1)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    risk, color = get_risk(prob)

    st.markdown(
        f"<h3 style='color:{color};text-align:center'>Prediction: {risk} ({prob:.2%})</h3>",
        unsafe_allow_html=True
    )
    st.write(f"Predicted Class: {prediction}")

    st.subheader("⚡ Top 3 Unusual Features for this Transaction")
    st.dataframe(top_features_df.style.format({"Deviation (z-score)": "{:.2f}"}))

# ===============================
# CHARTS
st.subheader("Fraud vs Non-Fraud Distribution")
fig, ax = plt.subplots(figsize=(5,5))
classes = data['Class'].value_counts()
ax.pie(
    classes,
    labels=["Not Fraud", "Fraud"],
    autopct="%1.1f%%",
    colors=["#87CEEB", "#FF4C4C"],
    shadow=True,
    startangle=90
)
ax.axis('equal')
st.pyplot(fig)

# ===============================
# DOWNLOAD TRANSACTION REPORT
st.subheader("📄 Download Transaction Report")
csv = transaction.to_frame().T.to_csv(index=False)
st.download_button(
    "Download CSV",
    csv,
    file_name=f"transaction_{row_index}.csv",
    mime="text/csv"
)

