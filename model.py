from keras.models import Sequential
from keras.optimizers import Adadelta, RMSprop, SGD
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np
from scipy.stats.stats import pearsonr
from keras.regularizers import *
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm




def build_model(datasize=20):
    # datasize = DATASIZE
    W_maxnorm = 3
    DROPOUT = 0.3  #{{choice([0.3, 0.5, 0.7])}}

    model = Sequential()
    model.add(Conv2D(128, (3, 5), border_mode='same', input_shape=(datasize, 4, 1), activation='relu')) # , W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPool2D(pool_size=(1, 5), strides=(1, 1),padding='same'))
    model.add(Conv2D(128, (3, 5), border_mode='same',
                     activation='relu'))  # , W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPool2D(pool_size=(1, 5), strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (5, 5), border_mode='same',
                     activation='relu'))  # , W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPool2D(pool_size=(1, 5), strides=(1, 1), padding='same'))
    # model.add(Convolution2D(256, 1, 5, border_mode='same', activation='relu', W_constraint=maxnorm(W_maxnorm)))
    # model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
    # model.add(Convolution2D(512, 1, 5, border_mode='same', activation='relu', W_constraint=maxnorm(W_maxnorm)))
    # model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(10, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(DROPOUT))
    model.add(Dense(1))
    # model.add(Activation('softmax'))

    myoptimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='mse', optimizer='adam')  #myoptimizer) #, metrics=['accuracy'])
    return model



def train(model, X_train, Y_train):
    # data_code = 'DATACODE'
    # topdir = 'TOPDIR'
    # model_arch = 'MODEL_ARCH'
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=5, validation_split=0.1)
    return model