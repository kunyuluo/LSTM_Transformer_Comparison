from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array


inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)

data = array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

pred = model.predict(data)
print(pred)
# print(inputs1)
