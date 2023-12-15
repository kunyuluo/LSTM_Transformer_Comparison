import pickle
from Helper import ETL, PredictAndForecast, Evaluate, plot_results


# Load the model
# *************************************************************************
model_index = 2
with open('models/model_lstm_{}.pkl'.format(model_index), 'rb') as f:
    model = pickle.load(f)

# Prepare the data
# *************************************************************************
data = ETL('AAPL')
# print(data.X_test.shape)
index = 4
sample_x = data.X_test[index].reshape(1, len(data.X_test[0]), 1)
sample_y = data.y_test[index].reshape(1, len(data.y_test[0]))

predict_values = PredictAndForecast(model, data.test).get_predictions()
# print(predict_values)
# print(predict_values.shape)

# Evaluate the prediction
# *************************************************************************
evals = Evaluate(data.test, predict_values)
print('Uni_LSTM Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))
print('Uni_LSTM Model\'s var ratio is: {}%'.format(round(evals.var_ratio*100, 1)))

# Visualize the results
# *************************************************************************
plot_results(data.test, predict_values, data.df)
