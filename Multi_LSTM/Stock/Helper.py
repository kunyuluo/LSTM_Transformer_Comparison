import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import pickle

data = pd.read_csv("train.csv", parse_dates=["Date"], index_col=[0])
# print(data.head())

test_split = round(len(data) * 0.20)

df_for_training = data[:-test_split]
df_for_testing = data[-test_split:]

scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = createXY(df_for_training_scaled, 30)

testX, testY = createXY(df_for_testing_scaled, 30)


def build_model(epochs=25, batch_size=32, x_test=testX, y_test=testY):
    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(30, 5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    history = grid_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
                             validation_data=(x_test, y_test), verbose=1)

    return grid_model, history


grid_model = build_model(epochs=25, batch_size=20)
# parameters = {'batch_size': [16, 20], 'epochs': [8, 10], 'optimizer': ['adam', 'Adadelta']}
# grid_search = GridSearchCV(estimator=grid_model, param_grid = parameters, cv = 2)
model = grid_model[0]
history = grid_model[1]

# Save models
# *************************************************************************
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
