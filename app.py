import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------------
# Files
MODEL_FILE = "trained_random_forest_model.pkl"
DATA_FILE = "creditcard.csv"
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Credit_card_font_awesome.svg/1200px-Credit_card_font_awesome.svg.png"

# -------------------------------
# Page config (MUST be first Streamlit call)
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Functions

@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)

@st.cache_data
def load_data(data_file):
    df = pd.read_csv(data_file)

    # Ensure required columns exist
    if "Time" not in df.columns:
        df.insert(0, "Time", 0)

    if "V1" not in df.columns:
        df.insert(1, "V1", 0)

    return df

@st.cache_data
def compute_probs(_model, X):
    return _model.predict_proba(X)[:, 1]

def pick_transaction(data, use_random, row_index):
    if use_random:
        row_index = random.randint(0, len(data) - 1)
    transaction = data.iloc[row_index]
    return transaction, row_index

def highlight_features(row, numeric_cols, full_data):
    styles = []
    for col in row.index:
        if col in numeric_cols:
            if row[col] > full_data[col].mean() + 3 * full_data[col].std():
                styles.append("color: red; font-weight: bold")
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

# -------------------------------
# Load model & data
model = load_model(MODEL_FILE)
data = load_data(DATA_FILE)

# 🔒 Ensure dataset matches model features
MODEL_FEATURES = model.feature_names_in_
data = data.reindex(columns=list(MODEL_FEATURES) + ["Class"], fill_value=0)

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Debug (you can remove later)
st.write("Dataset shape:", data.shape)

# -------------------------------
# Header
st.markdown(
    f"""
    <div style='background-color:#1E90FF;padding:15px;border-radius:10px;display:flex;align-items:center;'>
        <img src="{LOGO_URL}" width="50" style="margin-right:15px;">
        <div>
            <h1 style='color:white;margin:0;'>💳 Credit Card Fraud Detection App</h1>
            <p style='color:white;margin:0;'>Interactive Fraud Detection Dashboard</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar
row_index = 0  # IMPORTANT default

with st.sidebar:
    st.header("Options & Dataset Summary")
    use_random = st.checkbox("Pick a random transaction", value=True)

    if not use_random:
        row_index = st.number_input(
            f"Enter row index (0 to {len(data)-1})",
            min_value=0,
            max_value=len(data)-1,
            value=0,
            step=1
        )

    st.subheader("Dataset Stats")
    st.write(f"Total Transactions: {len(data):,}")
    st.write(f"Total Frauds: {int(data['Class'].sum()):,}")
    st.write(f"Total Non-Frauds: {len(data) - int(data['Class'].sum()):,}")
    st.write(f"Fraud Percentage: {data['Class'].mean()*100:.2f}%")

    # Top 5 High Risk
    st.subheader("🚨 Top 5 High-Risk Transactions")
    X = data.drop(columns=["Class"], errors="ignore")
    probs = compute_probs(model, X)

    data_with_prob = data.copy()
    data_with_prob["Fraud_Probability"] = probs

    top5 = data_with_prob.sort_values(
        by="Fraud_Probability", ascending=False
    ).head(5)

    st.dataframe(top5[["Fraud_Probability"]])

# -------------------------------
# Pick transaction
transaction, row_index = pick_transaction(
    data, use_random, row_index
)

# -------------------------------
# Display transaction
st.subheader(f"Selected Transaction (Row {row_index})")
st.dataframe(
    transaction.to_frame().T.style.apply(
        highlight_features,
        axis=1,
        numeric_cols=numeric_cols,
        full_data=data
    )
)

# -------------------------------
# Prediction
if st.button("Predict"):
    features = transaction.drop("Class").values.reshape(1, -1)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    risk, color = get_risk(prob)

    st.markdown(
        f"<h3 style='color:{color}'>Prediction: {risk} (Confidence: {prob:.2%})</h3>",
        unsafe_allow_html=True
    )
    st.write(f"🧾 Predicted Class: {prediction}")

    # Bar chart
    st.subheader("Transaction Amount Comparison")
    fig, ax = plt.subplots()
    ax.bar(
        ["Selected Transaction", "Average Transaction"],
        [transaction["Amount"], data["Amount"].mean()]
    )
    ax.set_ylabel("Amount ($)")
    st.pyplot(fig)

# -------------------------------
# Pie chart
st.subheader("Fraud vs Non-Fraud in Dataset")
fig2, ax2 = plt.subplots()
data["Class"].value_counts().plot.pie(
    autopct="%1.1f%%",
    startangle=90,
    ax=ax2
)
ax2.set_ylabel("")
st.pyplot(fig2)

# -------------------------------
# Download
st.subheader("📄 Download Transaction Report")
csv = transaction.to_frame().T.to_csv(index=False)
st.download_button(
    "Download CSV",
    csv,
    file_name=f"transaction_{row_index}.csv",
    mime="text/csv"
)


