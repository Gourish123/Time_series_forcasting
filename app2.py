import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

start='19-10-2020'
end= '16-10-2023'

st.title ('Stock Price Predictions:chart_with_upwards_trend:')

df = pd.read_csv('Company stock prices.csv')
df1=df.reset_index()['Close']

#Describing data
st.subheader('Stock Descriptive Stats for year 2020-2023')
st.write(df.describe())

#Visualization
st.subheader('Closing Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100Moving Average & 200Moving Average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    
#load my model

model=load_model('my_model2.keras')



import tensorflow as tf
tf.__version__
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting
# shift train predictions for plotting

look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
st.subheader('Train Predictions Vs TestPredictions')
fig3=plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
st.pyplot(fig3)



x_input=test_data[126:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):

    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


print(lst_output)


day_new=np.arange(1,101)
day_pred=np.arange(101,131)

import matplotlib.pyplot as plt
st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[653:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

df3=df1.tolist()
fig4=plt.figure(figsize=(12,6))
df3.extend(lst_output)
plt.plot(df3[653:],'r')
st.pyplot(fig4)


df4=scaler.inverse_transform(lst_output)

import datetime
import pandas as pd
 
# initializing date
test_date = datetime.datetime.strptime("17-10-2023", "%d-%m-%Y")
 
# initializing K 
K = 30
 
date_generated = pd.date_range(test_date, periods=K)
print(date_generated.strftime("%d-%m-%Y"))
 
date_generated = pd.date_range(test_date, periods=K)
print(date_generated.strftime("%d-%m-%Y"))
Final = pd.DataFrame(df4,index=date_generated, columns=['Predicted Price'])
st.table(Final)
