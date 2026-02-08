import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------------
MODEL_FILE = "trained_random_forest_model.pkl"
DATA_FILE = "creditcard.csv"
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Credit_card_font_awesome.svg/1200px-Credit_card_font_awesome.svg.png"

# -------------------------------
# -------------------------------
# Functions

@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)

@st.cache_data
def load_data(data_file):
    return pd.read_csv(data_file)

@st.cache_data
def compute_probs(_model, X):
    # _model is ignored by Streamlit for caching
    return _model.predict_proba(X)[:, 1]

def pick_transaction(data, use_random, row_index):
    if use_random:
        row_index = random.randint(0, len(data)-1)
    transaction = data.iloc[row_index]
    return transaction, row_index

def highlight_features(row, numeric_cols):
    styles = []
    for col in row.index:
        val = row[col]
        if col in numeric_cols:
            if val > data[col].mean() + 3*data[col].std():
                styles.append('color: red; font-weight: bold')
            else:
                styles.append('')
        else:
            styles.append('')
    return styles

def get_risk(prob):
    if prob >= 0.8:
        return "High Risk ⚠️", "#1E90FF"
    elif prob >= 0.5:
        return "Medium Risk ⚠️", "#75C3F4"
    else:
        return "Low Risk ✅", "#82D9E5"

# -------------------------------
# Load model and data

try:
    model = load_model(MODEL_FILE)
    
    # Try to load the real file; if it's missing (due to upload limits), generate dummy data
    try:
        data = load_data(DATA_FILE)
    except FileNotFoundError:
        st.warning("⚠️ Large dataset not found on server. Generating demo data for preview.")
        # Create columns matching the original dataset: Time, V1-V28, Amount, Class
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        data = pd.DataFrame(np.random.randn(100, 31), columns=cols)
        data['Class'] = np.random.randint(0, 2, 100)
        
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

numeric_cols = [col for col in data.select_dtypes(include=np.number).columns if col != 'Class']

# -------------------------------
# Streamlit App Config
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Header with logo
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
    
    # Dataset summary stats
    st.subheader("Dataset Stats")
    st.write(f"Total Transactions: {len(data):,}")
    st.write(f"Total Frauds: {data['Class'].sum():,}")
    st.write(f"Total Non-Frauds: {len(data) - data['Class'].sum():,}")
    st.write(f"Fraud Percentage: {data['Class'].mean()*100:.2f}%")
    
    # -------------------------------
    # Top 5 High-Risk Transactions
    st.subheader("🚨 Top 5 High-Risk Transactions")
    X = data.drop(columns=['Class'], errors='ignore')
    probs = compute_probs(model, X)
    data_with_prob = data.copy()
    data_with_prob['Fraud_Probability'] = probs
    top5 = data_with_prob.sort_values(by='Fraud_Probability', ascending=False).head(5)
    
    def highlight_top5(val):
        if val > 0.8:
            return 'background-color:#1E90FF; color:white; font-weight:bold'
        elif val > 0.5:
            return 'background-color:#87CEFA; color:black; font-weight:bold'
        else:
            return ''
    
    st.dataframe(top5[['Fraud_Probability']].style.applymap(highlight_top5))

# -------------------------------
# Pick transaction
transaction, row_index = pick_transaction(data, use_random, row_index if not use_random else None)

# -------------------------------
# Highlight unusual values
st.subheader(f"Selected Transaction (Row {row_index})")
st.dataframe(transaction.to_frame().T.style.apply(highlight_features, axis=1, numeric_cols=numeric_cols))

# -------------------------------
# Predict button
if st.button("Predict"):
    features = transaction.drop("Class", errors='ignore').values.reshape(1, -1)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    risk, color = get_risk(prob)
    
    st.markdown(f"<h3 style='color:{color}'>Prediction: {risk} (Confidence: {prob:.2%})</h3>", unsafe_allow_html=True)
    st.write(f"🧾 Predicted Class: {prediction}")

    # Bar chart: Transaction amount vs average
    st.subheader("Transaction Amount Comparison")
    amounts = [transaction['Amount'], data['Amount'].mean()]
    labels = ["Selected Transaction", "Average Transaction"]
    colors = ["#71C8EB", '#1E90FF']
    
    fig, ax = plt.subplots()
    ax.bar(labels, amounts, color=colors)
    ax.set_ylabel("Amount ($)")
    ax.set_title("Selected Transaction vs Average Transaction Amount")
    st.pyplot(fig)

# -------------------------------
# Pie chart: Dataset class distribution
st.subheader("Fraud vs Non-Fraud in Dataset")
fig2, ax2 = plt.subplots()
data['Class'].value_counts().plot.pie(
    labels=['Not Fraud (0)', 'Fraud (1)'],
    autopct='%1.1f%%',
    colors=['#87CEEB','#1E90FF'],
    startangle=90,
    ax=ax2
)
ax2.set_ylabel('')
st.pyplot(fig2)

# -------------------------------
# Download report button
st.subheader("📄 Download Transaction Report")
csv = transaction.to_frame().T.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"transaction_{row_index}_report.csv",
    mime="text/csv"
)

