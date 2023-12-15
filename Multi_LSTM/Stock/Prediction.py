import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the model
# *************************************************************************
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

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


testX, testY = createXY(df_for_testing_scaled, 30)
prediction = model.predict(testX)

prediction_copies_array = np.repeat(prediction, 5, axis=-1)
print(prediction_copies_array)
pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), 5)))
print(pred)

# original_copies_array = np.repeat(testY, 5, axis=-1)
# original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 5)))[:, 0]

# print("Pred Values-- ", pred)
# print("\nOriginal Values-- ", original)
#
# plt.plot(original, color='red', label='Real Stock Price')
# plt.plot(pred, color='blue', label='Predicted Stock Price')
# plt.title(' Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel(' Stock Price')
# plt.legend()
# plt.show()
