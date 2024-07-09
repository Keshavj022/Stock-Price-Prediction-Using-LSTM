import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_lstm_model():
    return load_model('stock_price_lstm_model.keras')

# Function to fetch stock data
def fetch_data(stock_symbol, start_date, end_date):
    return yf.download(stock_symbol, start=start_date, end=end_date)

# Function to create the dataset for predictions
def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
    return np.array(X)

# Title of the app
st.title("Stock Price Prediction using LSTM")

# Sidebar for user inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-01-01'))

# Load data
data = fetch_data(stock_symbol, start_date, end_date)
st.write(f"### {stock_symbol} Stock Price Data")
st.line_chart(data['Close'])

# Preprocessing the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Creating dataset for prediction
time_step = 100
X = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Loading the model
model = load_lstm_model()

# Making predictions
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Plotting predictions
plt.figure(figsize=(16, 8))
plt.plot(data['Close'].values, label='Actual Stock Price')
plt.plot(range(time_step, time_step + len(predictions)), predictions, label='Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)
