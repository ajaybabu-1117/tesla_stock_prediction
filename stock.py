import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
# --- Streamlit UI Setup ---
st.set_page_config(page_title="TSLA Stock Predictor", page_icon="📈", layout="centered")
st.title("📈 TSLA Stock Price Predictor")
st.markdown("Enter a future date to predict Tesla's stock price using a pre-trained LSTM model.")

# --- Load Model ---
MODEL_PATH = "C:/datasets/joblib models/tsla_lstm_model.h5"

@st.cache_resource
def load_lstm_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error("❌ Model file not found. Please train and save the LSTM model first.")
        return None

model = load_lstm_model()

# --- Load Data ---
@st.cache_data
def load_and_scale_data():
    df = yf.download('TSLA', start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    return df, scaled, scaler

df, scaled_data, scaler = load_and_scale_data()

# --- Prediction Logic ---
def predict_tsla_price(model, target_date_str):
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        last_date = df.index[-1].date()

        if target_date <= last_date:
            return f"⚠️ Please enter a **future date** after {last_date}", None

        # Business days from last known date to target date
        future_dates = pd.date_range(start=last_date + timedelta(days=1), end=target_date, freq='B')
        days_to_predict = len(future_dates)

        if days_to_predict == 0:
            return "⚠️ No trading days between now and the date entered.", None

        input_sequence = scaled_data[-60:].tolist()
        predictions_scaled = []

        for _ in range(days_to_predict):
            X_input = np.array(input_sequence[-60:]).reshape(1, 60, 1)
            pred_scaled = model.predict(X_input, verbose=0)[0][0]
            predictions_scaled.append(pred_scaled)
            input_sequence.append([pred_scaled])

        # Final prediction value
        final_price = scaler.inverse_transform(np.array([[predictions_scaled[-1]]]))[0][0]
        return None, round(final_price, 2)

    except ValueError:
        return "❌ Invalid date format. Use YYYY-MM-DD.", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# --- UI Input ---
st.subheader("📅 Enter Date for Prediction")
date_input = st.text_input("Enter a future date (YYYY-MM-DD)", placeholder="2025-06-01")

if st.button("🔮 Predict Price"):
    if not date_input:
        st.warning("Please enter a date.")
    elif model:
        msg, prediction = predict_tsla_price(model, date_input)
        if msg:
            st.warning(msg)
        else:
            st.success(f"💰 Predicted TSLA price on **{date_input}** is **${prediction}**")

st.markdown("---")
st.caption("🔧 Built by Ajay • Powered by LSTM, Streamlit, and yFinance")
