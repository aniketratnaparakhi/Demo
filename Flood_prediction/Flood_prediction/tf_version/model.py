# Importing the libraries
import numpy as np
import pandas as pd

dataset = pd.read_csv('inflow_data.csv')

X = dataset.iloc[:, 1:]

y = dataset.iloc[:, 0]

n_features = X.shape[1]

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_shape=(n_features,)))
model.add(Dense(1, activation='linear'))

# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# fit the model
model.fit(X,y, epochs=100, batch_size=32, verbose=2,shuffle=False)

# Saving model to disk
model.save("model.h5")

from tensorflow.keras.models import load_model

# Loading model to compare the results
model = load_model('model.h5')

print(model.predict([[15200,14300]]))