# Stock Market Prediction And Forecasting Using Stacked LSTM
from flask import Flask
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import requests
import time

app = Flask(__name__)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY
def new_dataset(dataset, step_size):
    data_X = []
    data_Y = []
    for i in range(len(dataset)-step_size-1):
        a = dataset[i:(i+step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

# Data obtained from NBP API 5 years back
start_date = date.today() - timedelta(days=(367 * 5))
end_date = date.today()
delta = timedelta(days=367)
daterange = []

while start_date < end_date:
    daterange.append(start_date.strftime("%Y-%m-%d"))
    start_date += delta
daterange.append(end_date.strftime("%Y-%m-%d"))

data_list_EUR = []
for x in range(len(daterange)):
    if x <= (len(daterange) - 2):
        try:
            print('Connecting... ' + str(x) + ' time')
            response_API = requests.get(
                "https://api.nbp.pl/api/exchangerates/rates/a/eur/" + daterange[x] + "/" + daterange[x + 1] + "?format=json")
            data = response_API.json()['rates']
            data_list_EUR.append(data)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
            print('Reconnecting...')
            time.sleep(5)
            response_API = requests.get(
                "https://api.nbp.pl/api/exchangerates/rates/a/eur/" + daterange[x] + "/" + daterange[x + 1] + "?format=json")
            data = response_API.json()['rates']
            data_list_EUR.append(data)

merged_data_list_EUR = list(itertools.chain(*data_list_EUR))
df_EUR = pd.DataFrame(merged_data_list_EUR)
df_EUR_1 = df_EUR.reset_index()['mid']

# LSTM are sensitive to the scale of the data. so we apply MinMax scaler
scaler_EUR = MinMaxScaler(feature_range=(0, 1))
df_EUR_1 = scaler_EUR.fit_transform(np.array(df_EUR_1).reshape(-1, 1))

# Splitting dataset into train and test split
training_size_EUR = int(len(df_EUR_1) * 0.65)
test_size_EUR = len(df_EUR_1) - training_size_EUR
train_data_EUR, test_data_EUR = df_EUR_1[0:training_size_EUR, :], df_EUR_1[training_size_EUR:len(df_EUR_1), :1]

# RESHAPING TRAIN AND TEST DATA
time_step_EUR = 100
X_train_EUR, y_train_EUR = new_dataset(train_data_EUR, time_step_EUR)
X_test_EUR, ytest_EUR = new_dataset(test_data_EUR, time_step_EUR)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train_EUR = X_train_EUR.reshape(X_train_EUR.shape[0], X_train_EUR.shape[1], 1)
X_test_EUR = X_test_EUR.reshape(X_test_EUR.shape[0], X_test_EUR.shape[1], 1)

# Create the Stacked LSTM model
model_EUR = Sequential()
model_EUR.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_EUR.add(LSTM(50, return_sequences=True))
model_EUR.add(LSTM(50))
model_EUR.add(Dense(1))
model_EUR.compile(loss='mean_squared_error', optimizer='adam')
model_EUR.summary()
model_EUR.fit(X_train_EUR, y_train_EUR, validation_data=(X_test_EUR, ytest_EUR), epochs=10, batch_size=64, verbose=1)

# Lets Do the prediction and check performance metrics
train_predict_EUR = model_EUR.predict(X_train_EUR)
test_predict_EUR = model_EUR.predict(X_test_EUR)

# Transformback to original form
train_predict_EUR = scaler_EUR.inverse_transform(train_predict_EUR)
test_predict_EUR = scaler_EUR.inverse_transform(test_predict_EUR)

# Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train_EUR, train_predict_EUR))

# Test Data RMSE
math.sqrt(mean_squared_error(ytest_EUR, test_predict_EUR))

print(len(test_data_EUR))  # 446

x_input_EUR = test_data_EUR[346:].reshape(1, -1)
temp_input_EUR = list(x_input_EUR)
temp_input_EUR = temp_input_EUR[0].tolist()

# Demonstrate prediction for next 30 days
lst_output_EUR = []
n_steps_EUR = 100
i_EUR = 0
while i_EUR < 30:
    if len(temp_input_EUR) > 100:
        # EUR
        x_input_EUR = np.array(temp_input_EUR[1:])
        x_input_EUR = x_input_EUR.reshape(1, -1)
        x_input_EUR = x_input_EUR.reshape((1, n_steps_EUR, 1))
        yhat_EUR = model_EUR.predict(x_input_EUR, verbose=0)
        temp_input_EUR.extend(yhat_EUR[0].tolist())
        temp_input_EUR = temp_input_EUR[1:]
        lst_output_EUR.extend(yhat_EUR.tolist())
        i_EUR = i_EUR + 1
    else:
        # EUR
        x_input_EUR = x_input_EUR.reshape((1, n_steps_EUR, 1))
        yhat_EUR = model_EUR.predict(x_input_EUR, verbose=0)
        temp_input_EUR.extend(yhat_EUR[0].tolist())
        lst_output_EUR.extend(yhat_EUR.tolist())
        i_EUR = i_EUR + 1

predictions_EUR = scaler_EUR.inverse_transform(lst_output_EUR).tolist()

@app.route('/')
def get_eur_pred():
    return predictions_EUR

if __name__ == "__main__":
    app.run(debug=True)
