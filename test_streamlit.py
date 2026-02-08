# test_streamlit.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("✅ Streamlit + Pandas + Scikit-learn Test")

# Sample data
data = {
    "Hours Studied": [1, 2, 3, 4, 5],
    "Score": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
st.write("Sample Data:", df)

# Simple Linear Regression
X = df[["Hours Studied"]]
y = df["Score"]
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[6]])

st.write(f"Predicted Score for 6 hours of study: {prediction[0]:.2f}")
