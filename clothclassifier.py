import tensorflow as tf
from tensorflow import keras
import numpy as np
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<.5):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True
mist = keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test) = mist.load_data()
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=keras.activations.relu),
    keras.layers.Dense(10,activation=keras.activations.softmax)
])
d = myCallback()
model.compile(optimizer=keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,callbacks=[d])
classification = model.predict(X_test)
print(classification[0])
print(y_test[0])

