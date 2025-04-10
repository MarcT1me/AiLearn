import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# other
import numpy as np  # math
import matplotlib.pyplot as plt  # deagrams and visualisation
from loguru import logger  # logging
# AI
import keras

logger.success('import np, plt and TensorFlow')

# data
shape = (1,)
c = [np.random.randint(-100, 500) for _ in range(1000)]  # input values
data = np.array(c)
f = [1.8*i + 32 for i in c]  # output values
layers = np.array(f)

# model himself
model = keras.Sequential()
# create layer
model.add(keras.layers.Dense(units=1, input_shape=shape, activation='linear'))
# create AI
logger.debug('compile model')
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
logger.success('compile model')

logger.debug('fit model')
history = model.fit(data, layers, epochs=500, batch_size=32)
logger.success('success fit')


def main():
    logger.debug(f'weights {tuple(model.get_weights())}')
    while True:
        degr = int(input('your degr:'))
        
        prompt = np.array(
            [*shape,
             degr]
        )
        print(f'out: {model.predict(prompt)[-1]}\tshould be {1.8*degr + 32}')


from threading import Thread

Thread(target=main).start()

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
