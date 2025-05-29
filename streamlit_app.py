# ==============================================================================
# Streamlit App for TSLA Stock Forecasting using CNN-LSTM Model
# Developed to provide interactive charting and multi-step forecasting
# using historical stock price data from Yahoo Finance.
#
# TUJUAN:
# - Mengambil data historis harga saham TSLA (Tesla Inc.) secara otomatis
# - Memvisualisasikan grafik harga saham TSLA
# - Melakukan prediksi harga saham ke depan (7-60 hari) menggunakan model CNN-LSTM
# - Menampilkan hasil evaluasi model terhadap data yang belum terlihat (data tahun 2025)
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import date, timedelta

# ------------------------------------------------------------------------------
# Konfigurasi Awal Aplikasi
# ------------------------------------------------------------------------------
st.set_page_config(page_title="📈 TSLA Live Chart Viewer", layout="centered")
st.title("📊 TSLA Interactive Stock Chart with Forecast")

# ------------------------------------------------------------------------------
# Sidebar - Input Rentang Tanggal dan Parameter Forecast
# ------------------------------------------------------------------------------
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Validasi input tanggal
if start_date >= end_date:
    st.error("❌ End date must be after start date.")
    st.stop()

st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=60, value=30)

# ------------------------------------------------------------------------------
# Fungsi untuk Mengambil Data dari Yahoo Finance (cached)
# ------------------------------------------------------------------------------
@st.cache_data
def load_data(start, end):
    df = yf.download('TSLA', start=start, end=end)
    df = df[['Close']].dropna()  # Hanya kolom harga penutupan yang digunakan
    return df

# Ambil data dengan loading spinner
with st.spinner("📥 Downloading TSLA data..."):
    df = load_data(start_date, end_date)

# Validasi jumlah data cukup untuk input model
if df.empty or len(df) < 60:
    st.warning("⚠️ Data kosong atau tidak cukup untuk forecast.")
    st.stop()

# ------------------------------------------------------------------------------
# Load Model CNN-LSTM dan Scaler (MinMaxScaler)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = load_model("cnn_lstm_best_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')
    scaler = joblib.load("cnn_lstm_scaler.save")
    return model, scaler

model, scaler = load_model_and_scaler()

# ------------------------------------------------------------------------------
# Fungsi untuk Forecast Multi-step Menggunakan Sliding Window (Autoregressive)
# ------------------------------------------------------------------------------
def direct_multistep_forecast(model, last_window, steps):
    forecast = []
    input_seq = last_window.copy()
    for _ in range(steps):
        pred = model.predict(input_seq.reshape(1, input_seq.shape[0], 1), verbose=0)
        forecast.append(pred[0, 0])
        # Update input: buang yang paling lama dan tambahkan prediksi terbaru
        input_seq = np.vstack([input_seq[1:], [[pred[0, 0]]]])
    return forecast

# ------------------------------------------------------------------------------
# Proses Forecasting
# ------------------------------------------------------------------------------
scaled = scaler.transform(df[['Close']])  # Normalisasi data
last_window = scaled[-60:]  # Gunakan 60 data terakhir sebagai input
forecast_scaled = direct_multistep_forecast(model, last_window, forecast_days)
forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

# Buat tanggal untuk hasil forecast (hari kerja)
forecast_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_prices})

# ------------------------------------------------------------------------------
# Evaluasi Model terhadap Data Tak Terlihat (setelah 2024-12-31)
# ------------------------------------------------------------------------------
eval_df = df[df.index > '2024-12-31']
scaled_eval = scaler.transform(eval_df[['Close']]) if not eval_df.empty else None

if scaled_eval is not None and len(scaled_eval) > 60:
    X_eval = []
    y_eval = []
    for i in range(60, len(scaled_eval)):
        X_eval.append(scaled_eval[i-60:i])
        y_eval.append(scaled_eval[i][0])
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)

    # Prediksi dan inverse scaling
    y_pred = model.predict(X_eval).flatten()
    y_test_inv = scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Hitung metrik evaluasi
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100

    # Tampilkan metrik evaluasi
    st.subheader("📐 Model Evaluation on Unseen 2025 Data")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f} USD")
    col2.metric("MAPE", f"{mape:.2f} %")

    eval_start = eval_df.index[60]
    eval_end = eval_df.index[-1]
    st.caption(f"Evaluated on unseen data from {eval_start.strftime('%Y-%m-%d')} to {eval_end.strftime('%Y-%m-%d')}")

# ------------------------------------------------------------------------------
# Visualisasi Harga Historis dan Hasil Forecast
# ------------------------------------------------------------------------------
st.subheader("📈 TSLA Close Price Chart + Forecast")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['Close'], marker='o', linestyle='-', color='blue', label='Historical Close')
ax.plot(forecast_df['Date'], forecast_df['Forecast'], linestyle='--', color='green', label='Forecast')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title("TSLA Close Price with Forecast")
ax.legend()
ax.grid(True)
fig.autofmt_xdate()

st.pyplot(fig)

# ------------------------------------------------------------------------------
# Tampilan Data Mentah dan Forecast (Opsional)
# ------------------------------------------------------------------------------
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

with st.expander("🔮 Show Forecast Data"):
    st.dataframe(forecast_df)

# ------------------------------------------------------------------------------
# Ringkasan Forecast
# ------------------------------------------------------------------------------
st.subheader("📝 Forecast Summary")
start_price = float(df['Close'].iloc[-1])
end_price = float(forecast_prices[-1])
change = float(end_price - start_price)
pct_change = (change / start_price) * 100

# Tentukan tren berdasarkan perubahan harga
trend = "📈 Expected to Increase" if change > 0 else "📉 Expected to Decrease" if change < 0 else "⏸ No Significant Change"

# Tampilkan ringkasan forecast
st.markdown(f"From {forecast_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {forecast_df['Date'].iloc[-1].strftime('%Y-%m-%d')}:")
st.markdown(f"- Start Price: ${start_price:.2f}")
st.markdown(f"- End Forecast Price: ${end_price:.2f}")
st.markdown(f"- Change: {change:.2f} USD ({pct_change:.2f}%)")
st.markdown(f"- Trend: {trend}")
