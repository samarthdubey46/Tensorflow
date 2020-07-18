import tensorflow as tf
import numpy as np
from tensorflow import keras
def house_model(i):
    model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    ys = np.array([1, 1.5, 2, 2.5, 3, 3.5], dtype=float)
    model.compile(optimizer='sgd',loss="mean_squared_error")
    model.fit(xs,ys,epochs=500)
    return model.predict(i)[0]
print(house_model([7.0]))