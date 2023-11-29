import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras import layers
from Helper import ETL
from Visualization import plot_results


class PredictAndForecast:
    """
    model: tf.keras.Model
    train: np.array
    test: np.array
    Takes a trained model, train, and test datasets and returns predictions
    of len(test) with same shape.
    """
    def __init__(self, model, train, test, n_input=5) -> None:
        self.model = model
        self.train = train
        self.test = test
        self.n_input = n_input
        self.predictions = self.get_predictions()

    def forcast(self, history) -> np.array:
        """
        Given last weeks actual data, forecasts next weeks prices.
        """
        # Flatten data
        data = np.array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

        # retrieve last observations for input data
        x_input = data[-self.n_input:, :]
        x_input = x_input.reshape((1, len(x_input), 1))

        # forecast the next week
        yhat = self.model.predict(x_input, verbose=0)

        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def get_predictions(self) -> np.array:
        """
        compiles models predictions week by week over entire
        test set.
        """
        # history is a list of weekly data
        history = [x for x in self.train]

        # walk-forward validation over each week
        predictions = []
        for i in range(len(self.test)):
            yhat_sequence = self.forcast(history)

            # store the predictions
            predictions.append(yhat_sequence)

            # get real observation and add to history for predicting the next week
            history.append(self.test[i, :])

        return np.array(predictions)


class Evaluate:
    def __init__(self, actual, predictions) -> None:
        self.actual = actual
        self.predictions = predictions
        self.var_ratio = self.compare_var()
        self.mape = self.evaluate_model_with_mape()

    def compare_var(self) -> float:
        """
        Calculates the variance ratio of the predictions
        """
        return abs(1 - (np.var(self.predictions)) / np.var(self.actual))

    def evaluate_model_with_mape(self) -> float:
        """
        Calculates the mean absolute percentage error
        """
        return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())


def build_lstm(etl: ETL, epochs=25, batch_size=32):
    """
      Builds, compiles, and fits our LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = 5, 1, 5
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs))
    print("compliling baseline model")

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    print("fitting model")

    history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs, validation_data=(etl.X_test, etl.y_test))

    return model, history


data = ETL('AAPL')
baseline = build_lstm(data)
baseline_model = baseline[0]
history = baseline[1]
baseline_preds = PredictAndForecast(baseline_model, data.train, data.test)
baseline_evals = Evaluate(data.test, baseline_preds.predictions)
plot_results(data.test, baseline_preds.predictions, data.df, title_suffix='LSTM')
