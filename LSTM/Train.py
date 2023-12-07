from Helper import ETL, plot_metrics
from LSTM import build_lstm_1, build_lstm_2, build_lstm_3
import pickle

# Prepare the data
# *************************************************************************
data = ETL('AAPL')

# Build the model
# *************************************************************************
model_index = 1

if model_index == 1:
    baseline = build_lstm_1(data)
elif model_index == 2:
    baseline = build_lstm_2(data)
else:
    baseline = build_lstm_3(data)

model = baseline[0]
history = baseline[1]

# Check metrics
# *************************************************************************
plot_metrics(history)

# Save models
# *************************************************************************
with open('model_lstm_{}.pkl'.format(model_index), 'wb') as f:
    pickle.dump(model, f)
