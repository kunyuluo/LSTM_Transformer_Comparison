import pandas as pd
import numpy as np
import yfinance as yf

# Load the stock price data of Apple
# ********************************************************************************
t = yf.Ticker('AAPL')
history = t.history(period='max')
data = history['Close']
# history.to_csv('AAPL.csv')
# print(data)
# print(len(data))

# Split the data into train and test set
# ********************************************************************************
test_size = 0.2

train_index = round(len(data) * (1 - test_size))
train = np.array(data[0:train_index])
test = np.array(data[train_index:])
# print(train.shape)

# reshape the data
# ********************************************************************************
timestep = 5
num_features = 1

train_remainder = len(train) % timestep
test_remainder = len(test) % timestep
if train_remainder != 0 and test_remainder != 0:
    train = train[train_remainder:]
    test = test[test_remainder:]
elif train_remainder != 0:
    train = train[train_remainder:]
elif test_remainder != 0:
    test = test[test_remainder:]

sample_train = int(len(train) / timestep)
sample_test = int(len(test) / timestep)
train = np.array(np.array_split(train, sample_train))
test = np.array(np.array_split(test, sample_test))

train = train.reshape((sample_train, timestep, num_features))
test = test.reshape((sample_test, timestep, num_features))


# Construct dataset: x_train, y_train, x_test, y_test
# ********************************************************************************
n_input = 5

train = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
sample = train[0:5, 0]
sample = sample.reshape((len(sample), 1))
print(sample)
print(sample.shape)


def to_supervised(train, n_out=5) -> tuple:
    """
    Converts our time series prediction problem to a
    supervised learning problem.
    """
    # flatted the data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
    return np.array(X), np.array(y)


# x_train, y_train = to_supervised(train)
# x_test, y_test = to_supervised(test)

# data_check = np.array(train)
# data_check = data_check.reshape(data_check.shape[0] * data_check.shape[1], data_check.shape[2])
# x_input = data_check[-5:, :]
# x_input = x_input.reshape((1, len(x_input), 1))
#
# week_data = [x for x in train]
# print(week_data)

# print(x_train)
# print(y_train)