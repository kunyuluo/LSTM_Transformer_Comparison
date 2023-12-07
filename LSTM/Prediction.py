import pickle
from Helper import ETL, PredictAndForecast, Evaluate, plot_results


# Load the model
# *************************************************************************
with open('models/model_lstm_1.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the data
# *************************************************************************
data = ETL('AAPL')
# print(data.df)

predict_values = PredictAndForecast(model, data.train, data.test).get_predictions()
# print(predict_values)
# print(predict_values.shape)

# Evaluate the prediction
# *************************************************************************
evals = Evaluate(data.test, predict_values)
print('LSTM Model\'s mape is: {}'.format(evals.mape))
print('LSTM Model\'s var ratio is: {}'.format(evals.var_ratio))

# Visualize the results
# *************************************************************************
plot_results(data.test, predict_values, data.df)
