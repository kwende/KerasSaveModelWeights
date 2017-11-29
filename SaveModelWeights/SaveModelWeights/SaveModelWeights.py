
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
        print('epoch end')
        #print(model.get_weights())
        ##model.save_weights('c:/users/brush/desktop/weights.h5')
        #global lastEpoch
        #current = logs.get("loss")         
        #if current != None and current < self.value:
        #    self.model.stop_training = True
        #    lastEpoch = epoch + 1

x = np.array([
    [0,0], 
    [0,1],
    [1,0], 
    [1,1]
])
y = np.array([
    [0], 
    [1], 
    [1], 
    [0]
])


model.add(Dense(units=3, 
                input_dim = 2, 
                use_bias = True, 
                kernel_initializer = VarianceScaling()))
model.add(Dense(units=1, 
                use_bias = True, 
                kernel_initializer = VarianceScaling()))
model.add(Activation('sigmoid'))
model.compile(loss = "mean_squared_error", 
              optimizer = SGD(lr = 0.5))

model.fit(x, y, 
          verbose = 1, 
          batch_size = 4, 
          epochs = 100, 
          callbacks = [
            EarlyStoppingByLossVal()
          ])

print(model.predict_proba(x))
print("Last epoch: " + str(lastEpoch))