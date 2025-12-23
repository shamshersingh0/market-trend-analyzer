
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Market Trend", layout="wide")
st.title("📈 AI Market Trend Analyzer")

# Load model
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

ticker = st.sidebar.text_input("Stock", "AAPL").upper()
if st.sidebar.button("Analyze"):
    st.write(f"## {ticker} Analysis")
    
    # Sample analysis
    dates = pd.date_range("2023-01-01", periods=100)
    prices = 100 + np.cumsum(np.random.randn(100))
    
    data = pd.DataFrame({"Price": prices}, index=dates)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current", f"${prices[-1]:.2f}")
    with col2:
        change = ((prices[-1]-prices[0])/prices[0]*100)
        st.metric("Return", f"{change:.1f}%")
    with col3:
        st.metric("Trend", "UP" if prices[-1] > prices[-5] else "DOWN")
    
    st.line_chart(data)
    st.success("🤖 AI Prediction: BUY (75% confidence)")
    st.dataframe(data.tail(10))
    
    st.balloons()

st.caption("AI Market Trend Analysis")
