import streamlit as st
from datetime import date
import yfinance as yf
from keras import models  
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

# Start and Ending Date
START = '2010-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

#Stocks to select from
stocks = ("AAPL", "GOOG", "MSFT", "GME")
# user_input = st.text_input("Enter Stock Ticker: ", 'AAPL')
selected_stock = st.selectbox("Select dataset for Prediction", stocks)

#Application Heading
st.title("Stock Prediction Application")



@st.cache_data
def load_data(ticker):
    data= yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data....")
data = load_data(selected_stock)
data_load_state.text("Loading Data.... Done!!")

st.subheader("RAW Data from 2010 - Today")
st.write(data.tail())
st.subheader("Generic Details of Data")
st.write(data.describe())

#-------------------------------------------------------
st.subheader("Closing Price vs Time Chart")

# Create a Plotly figure
fig = go.Figure()

# Add a trace for the closing prices
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))

# Customize the layout
fig.update_layout(
    title='Closing Price vs Time',
    xaxis_title='Time',
    yaxis_title='Closing Price',
    template='plotly_dark',  # Optional: Choose a theme
    autosize=True,
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)
#-------------------------------------------------------
st.subheader("Closing Price vs Time chart with 100 Days Moving Avg")
ma100 = data.Close.rolling(100).mean()

# Create the figure
fig = go.Figure()

# Add trace for closing prices
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='lines',
    name='Closing Price'
))

# Add trace for 100-day moving average
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=ma100,
    mode='lines',
    name='100-day MA',
    line=dict(color='red')
))

# Update layout for better visualization
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_dark',
    width=800,
    height=400,
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
#--------------------------------------------------


#Forecasting
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#model
model = models.load_model('keras_lstm_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing])
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])



x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Prediction
st.subheader('Predictions vs Original Closing Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b')
plt.plot(y_predicted, 'r')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig2)


df_train1 = data[['Date', 'Close']]
df_train1 = df_train1.rename(columns = {"Date": "ds", "Close": "y"})
n_years = st.slider("Years of Prediction: ", 1,4)
period = n_years*365
m = Prophet()
m.fit(df_train1)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast Data')
fig4 = plot_plotly(m, forecast)
st.plotly_chart(fig4)

st.write('Forecast Components')
fig5 = m.plot_components(forecast)
st.write(fig5)