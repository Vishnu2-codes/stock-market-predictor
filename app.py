import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
 

model = load_model('C:/Users/vishn/OneDrive/Desktop/Stock market predictor/stock_predictor.keras')


st.header('Stock Market Predictor')

stock = st.text_input('Enter stock symbol', 'GOOG')
start ='2012-01-01'
end ='2023-12-31'

data= yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)


data_train= pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test= pd.DataFrame(data.Close[int(len(data)*0.80):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))

pas_100_days= data_train.tail(100)
data_set= pd.concat([pas_100_days,data_test], ignore_index=True)
data_test_scale= scaler.fit_transform(data_test)

st.subheader('Price (Green) vs MA50(Red)')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price(green) vs MA50(Red) vs MA100(Blue)')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price(Green) vs MA100(Red) vs MA200(Blue)')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range  (100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0]) 
x,y= np.array(x),np.array(y) 
predict = model.predict(x)
scale=1/scaler.scale_
predict = predict * scale
y=y*scale

st.subheader('Orignal Price (Green) vs Predicted Price (Red)')
fig4 = plt.figure(figsize=(10,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
