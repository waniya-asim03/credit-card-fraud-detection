# 🛡️ AI Fraud Command Center

### A Professional-Grade Credit Card Fraud Detection & Explainable AI Dashboard

The **AI Fraud Command Center** transforms a traditional machine learning model into a real-time investigative dashboard for fraud analysts.

It combines a **Random Forest Classifier** with **Explainable AI (SHAP)** to not only predict fraudulent transactions but also clearly explain *why* a transaction is flagged as high-risk.

---

## 🚀 Live Demo

🔗 **(https://credit-card-fraud-detection-w9v5ihan3004sh.streamlit.app/)**

---

## ✨ Key Features

### 🔎 Forensic Analysis Engine

Instantly evaluate any transaction and receive:

* Fraud probability score
* Risk classification (Low / High Risk)
* Model confidence

---

### 🧠 Explainable AI with SHAP

Uses **SHapley Additive exPlanations (SHAP)** to:

* Break down the model’s prediction
* Highlight the top contributing features (e.g., V14, V17)
* Visualize feature impact using SHAP Waterfall plots

This ensures transparency and trust — critical in financial systems.

---

### 📊 Visual Outlier Signature (Radar Chart)

A Plotly-powered radar chart compares the selected transaction against dataset averages, allowing investigators to visually inspect abnormal behavior patterns.

---

### 🎚️ Interactive Risk Thresholding

Dynamic threshold slider to adjust fraud sensitivity in real time.
This simulates real-world trade-offs between:

* Fraud prevention
* Customer friction

---

### 🛡️ Robust Data Handling

Built-in validation mechanisms ensure:

* Schema alignment
* Missing feature handling
* Stable predictions during deployment

---

## 📊 Model Performance

* **Algorithm:** Random Forest Classifier
* **Problem Type:** Binary Classification
* **Dataset:** Highly imbalanced credit card fraud dataset (~0.17% fraud cases)
* **Evaluation Metrics:** Accuracy, Precision, Recall, ROC-AUC

The model is optimized to handle extreme class imbalance, prioritizing fraud detection sensitivity.

---

## 🛠️ Tech Stack

**Core:**

* Python
* Pandas
* NumPy

**Machine Learning:**

* Scikit-learn (Random Forest)

**Explainability:**

* SHAP

**Web Framework:**

* Streamlit

**Visualization:**

* Plotly
* Matplotlib

---

## 📂 Project Structure

```
ai-fraud-command-center/
│
├── app.py                          # Main Streamlit dashboard
├── trained_random_forest_model.pkl # Pre-trained ML model
├── creditcard_small.csv            # Dataset sample
├── requirements.txt                # Dependencies
├── runtime.txt                     # Deployment runtime
└── README.md                       # Documentation
```

---

## ▶️ Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-fraud-command-center.git
cd ai-fraud-command-center
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch Dashboard

```bash
streamlit run app.py
```

---

## 🔬 The Science Behind It

### Why Random Forest?

Random Forest is a powerful ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting. It performs particularly well in structured tabular data problems like fraud detection.

---

### Why Explainability Matters

In financial systems, black-box predictions are unacceptable.

By integrating SHAP:

* Each prediction is decomposed into feature contributions
* Auditors can verify the AI’s reasoning
* Regulatory transparency is supported

This bridges the gap between machine learning and real-world financial compliance.

---

## 👩‍💻 Author

**Waniya Asim**
Machine Learning & Data Science Enthusiast
