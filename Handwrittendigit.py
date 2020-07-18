
import matplotlib.pyplot as plt
import tensorflow.keras as keras
class callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= .9):
            print("Reached 90% Acurracy")
            self.model.stop_training = True
mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(units=512,activation='relu'),
    keras.layers.Dense(units=10,activation=keras.activations.softmax)
])
model.compile(optimizer='adam',loss=keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
c = callback()
model.fit(x_train,y_train,epochs=10,callbacks=[c])
c = model.predict(x_test)
