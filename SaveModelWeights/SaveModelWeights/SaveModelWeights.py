
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.initializers import VarianceScaling 
import numpy as np 


lastEpoch = 0
model = Sequential()


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.008, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        print(model.get_weights())
        raw_input()

np.random.seed(100)

model = Sequential()
model.add(Dense(input_dim=2, units=5))
model.add(Activation('sigmoid'))
model.add(Dense(input_dim=5, units=1))
model.add(Activation('sigmoid'))
X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y = np.array([[0],[1],[1],[0]], "float32")
sgd = SGD(lr=0.5)
model.compile(loss='binary_crossentropy', optimizer=sgd)
print(model.get_weights())
raw_input()
model.fit(X, y, nb_epoch=250, batch_size=1, callbacks = [
            EarlyStoppingByLossVal()
          ])

print(model.predict_proba(X))