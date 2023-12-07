import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


class ETL:
    def __init__(self, ticker, test_size=0.2, period='max', n_input=5, timestep=5) -> None:
        self.ticker = ticker
        self.period = period
        self.test_size = test_size
        self.n_input = n_input
        self.timestep = timestep
        self.df = self.extract_historic_data()
        self.train, self.test = self.etl()
        self.X_train, self.y_train = self.to_supervised(self.train)
        self.X_test, self.y_test = self.to_supervised(self.test)

    def extract_historic_data(self) -> pd.Series:
        """
        gets historical data from yf api.
        """
        t = yf.Ticker(self.ticker)
        history = t.history(period=self.period)
        return history.Close

    def split_data(self) -> tuple:
        """
        Splits our pd.Series into train and test series with
        test series representing test_size * 100 % of data.
        """
        data = self.extract_historic_data()
        if len(data) != 0:
            train_idx = round(len(data) * (1 - self.test_size))
            train = data[:train_idx]
            test = data[train_idx:]
            train = np.array(train)
            test = np.array(test)
            return train[:, np.newaxis], test[:, np.newaxis]
        else:
            raise Exception('Data set is empty, cannot split.')

    def window_and_reshape(self, data) -> np.array:
        """
        Reformats data into shape our model needs,
        namely, [# samples, timestep, # feautures]
        samples
        """
        NUM_FEATURES = 1
        samples = int(data.shape[0] / self.timestep)
        result = np.array(np.array_split(data, samples))
        return result.reshape((samples, self.timestep, NUM_FEATURES))

    def transform(self, train, test) -> np.array:
        train_remainder = train.shape[0] % self.timestep
        test_remainder = test.shape[0] % self.timestep
        if train_remainder != 0 and test_remainder != 0:
            train = train[train_remainder:]
            test = test[test_remainder:]
        elif train_remainder != 0:
            train = train[train_remainder:]
        elif test_remainder != 0:
            test = test[test_remainder:]
        return self.window_and_reshape(train), self.window_and_reshape(test)

    def etl(self) -> tuple[np.array, np.array]:
        """
        Runs complete ETL
        """
        train, test = self.split_data()
        return self.transform(train, test)

    def to_supervised(self, train, n_out=5) -> tuple:
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
            in_end = in_start + self.n_input
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
        Given last weeks actual data, forecasts next weeks' prices.
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
        compiles models predictions week by week over entire test set.
        """
        # history is a list of weekly data
        # history = [x for x in self.train]
        history = []

        # walk-forward validation over each week
        predictions = []
        for i in range(len(self.test)):
            # get real observation and add to history for predicting the next week
            history.append(self.test[i, :])

            yhat_sequence = self.forcast(history)

            # store the predictions
            predictions.append(yhat_sequence)

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


def plot_metrics(history, epochs: int = 25):
    acc = history.history['mape']
    val_acc = history.history['val_mape']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training MAPE')
    plt.plot(epochs_range, val_acc, label='Validation MAPE')
    plt.legend(loc='lower right')
    plt.title('Training and Validation MAPE')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_results(test, preds, df, title_suffix=None, xlabel='AAPL stock Price'):
    """
    Plots training data in blue, actual values in red, and predictions in green, over time.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    # x = df.Close[-498:].index
    plot_test = test[0:]
    plot_preds = preds[0:]

    x = df[-(plot_test.shape[0] * plot_test.shape[1]):].index
    plot_test = plot_test.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
    plot_preds = plot_preds.reshape((plot_test.shape[0] * plot_test.shape[1], 1))

    ax.plot(x, plot_test, label='actual')
    ax.plot(x, plot_preds, label='preds')

    if title_suffix is None:
        ax.set_title('Predictions vs. Actual')
    else:
        ax.set_title(f'Predictions vs. Actual, {title_suffix}')

    ax.set_xlabel('Date')
    ax.set_ylabel(xlabel)
    ax.legend()

    plt.show()
