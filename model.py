from keras.models import Sequential
from keras.optimizers import Adadelta, RMSprop, SGD
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
import numpy as np
from scipy.stats.stats import pearsonr
from keras.regularizers import *
# from keras.callbacks import TensorBoard
from keras.constraints import maxnorm
from keras.utils import plot_model
# from tensorboard._vendor.bleach import callbacks


def build_model(datasize=36):
    """
    This function builds the actual network architecture
    :param datasize: the input size of the selex sequence.
    :return:
    """
    W_maxnorm = 3

    model = Sequential()
    model.add(Conv2D(32, (6, 4), padding='same', input_shape=(datasize, 4, 1), activation='relu',
                     kernel_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPool2D(pool_size=(5, 1), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, (8, 4), padding='valid', input_shape=(datasize, 4, 1), activation='relu',
                     kernel_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPool2D(pool_size=(5, 1), strides=(1, 1), padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    # model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def train(model, X_train, Y_train, debug=True):
    """
    Train the network with the data
    :param model: the model variable
    :param X_train: selex sequences
    :param Y_train: label of each sequence
    :param debug: turn verbose on or off. Also possible to show on TesnorBoard
    :return:
    """
    if debug:
        import time
        log_dir = './Graph/'+str(time.time())
        tbCallBack = None  # TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, write_grads=True)
        cbk = [tbCallBack]
        verb = 1
    else:
        verb = 0
        cbk = None
    history = model.fit(X_train, Y_train, batch_size=512, epochs=10, validation_split=0.3, shuffle=True,
                            callbacks=cbk, verbose=verb)
    return model, history



def save_network(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    model.save('entire_model.h5')


def load_model(model):
    model.load_weights('model.h5')
    return model

def load_entire_model():
    from keras.models import model_from_json
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("entire_model.h5")
    return loaded_model


def predict(model, X_test):
    return model.predict(X_test)


def visualize_model(model):
    plot_model(model, to_file='model.png')
