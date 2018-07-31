from keras.models import Sequential
from keras.optimizers import Adadelta, RMSprop, SGD, adam
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
import numpy as np
from scipy.stats.stats import pearsonr
from keras.regularizers import *
from keras.callbacks import TensorBoard
from keras.constraints import maxnorm
from keras.utils import plot_model
from tensorboard._vendor.bleach import callbacks
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation, concatenate
from keras.initializers import normal



def build_model(datasize=36):
    # datasize = DATASIZE
    W_maxnorm = 3
    DROPOUT = 0.5  # {{choice([0.3, 0.5, 0.7])}}

    input_img = Input(shape=(datasize, 4, 1))
    tower_1 = Conv2D(32, (3, 4), padding='same', activation='relu')(input_img)
    tower_1 = MaxPool2D((3, 4), strides=(1, 1), padding='same')(tower_1)
    # tower_2 = Conv2D(8, (10, 4), padding='same', activation='relu')(input_img)
    # tower_2 = MaxPool2D((10, 4), strides=(1, 1), padding='same')(tower_2)
    tower_3 = Conv2D(32, (5, 4), padding='same', activation='relu')(input_img)
    tower_3 = MaxPool2D((3, 4), strides=(1, 1), padding='same')(tower_3)

    output = concatenate([tower_1, tower_3], axis=3)
    # output = Conv2D(32, (3, 4), padding='valid', activation='relu')(output)
    # output = MaxPool2D((3, 1), strides=(1, 1), padding='valid')(output)
    output = Conv2D(32, (5, 4), padding='valid', activation='relu')(output)
    output = MaxPool2D((3, 1), strides=(1, 1), padding='valid')(output)
    output = Flatten()(output)
    output = Dense(32, activation='relu')(output)
    out = Dense(5, activation='softmax')(output)
    model = Model(inputs=input_img, outputs=out)

    myoptimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    return model



def train(model, X_train, Y_train):
    import time
    log_name = './Graph/' + str(time.time())
    tbCallBack = TensorBoard(log_dir=log_name, histogram_freq=1, write_graph=True, write_images=True, write_grads=True)
    history = model.fit(X_train, Y_train, batch_size=512, epochs=12, validation_split=0.2, shuffle=True, callbacks=[tbCallBack])
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
    # model = build_model()
    return model.predict(X_test)


def visualize_model(model):
    plot_model(model, to_file='model.png')
    weights, biases = model.layers[0].get_weights()
    # print(weights.shape)
    # import os
    # for i in range(128):
    #     fname = 'filters/filter'+str(i) + '.csv'
    #     np.savetxt(fname, weights[:, :, 0, i], fmt='%.3f', newline=os.linesep)
