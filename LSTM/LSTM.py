import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import layers
from keras.optimizers import Adam
from Helper import ETL


def build_lstm_1(etl: ETL, epochs=25, batch_size=32):
    """
      Builds, compiles, and fits our LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = 5, 1, 5
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(230, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs))

    print("compliling baseline model")
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

    print("fitting model")
    history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(etl.X_test, etl.y_test), verbose=1)

    return model, history


def build_lstm_2(etl: ETL, epochs=25, batch_size=32):
    """
    Builds, compiles, and fits our LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = 5, 1, 5
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    # opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(etl.X_train.shape[1], etl.X_train.shape[2])))
    model.add(LSTM(64, activation='relu'))
    # model.add(LSTM(16))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs))

    print('compiling baseline model...')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    print('fitting model...')
    history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(etl.X_test, etl.y_test), verbose=1, callbacks=callbacks)

    return model, history


def build_lstm_3(etl: ETL, epochs=25, batch_size=48):
    """
    Builds, compiles, and fits our LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = 5, 1, 5
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=64))
    model.add(Dropout(0.1))
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(n_outputs))

    print('compiling baseline model...')
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

    print('fitting model...')
    history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(etl.X_test, etl.y_test), verbose=1, callbacks=callbacks)

    return model, history

# data = ETL('AAPL')
# baseline = build_lstm(data)
# baseline_model = baseline[0]
# history = baseline[1]
# baseline_preds = PredictAndForecast(baseline_model, data.train, data.test)
# baseline_evals = Evaluate(data.test, baseline_preds.predictions)
# plot_results(data.test, baseline_preds.predictions, data.df, title_suffix='LSTM')
