# from keras.models import Sequential
from keras.optimizers import Adadelta, RMSprop, SGD, adam
import numpy as np
# from scipy.stats.stats import pearsonr
# from keras.regularizers import *
from keras.callbacks import TensorBoard
from keras.constraints import maxnorm
from keras.utils import plot_model
# from tensorboard._vendor.bleach import callbacks
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, concatenate, MaxPooling2D, AveragePooling2D
from keras.initializers import normal
from keras.layers.normalization import BatchNormalization
# from keras import regularizers

np.random.seed(1337) # for reproducibility


# def build_model(datasize=36):
#     # datasize = DATASIZE
#     W_maxnorm = 3
#     DROPOUT = 0.5  # {{choice([0.3, 0.5, 0.7])}}
#
#     input_img = Input(shape=(datasize, 4, 1))
#     tower_1 = Conv2D(32, (3, 4), padding='same', activation='relu')(input_img)
#     tower_1 = MaxPool2D((3, 4), strides=(1, 1), padding='same')(tower_1)
#     # tower_2 = Conv2D(8, (10, 4), padding='same', activation='relu')(input_img)
#     # tower_2 = MaxPool2D((10, 4), strides=(1, 1), padding='same')(tower_2)
#     tower_3 = Conv2D(32, (5, 4), padding='same', activation='relu')(input_img)
#     tower_3 = MaxPool2D((3, 4), strides=(1, 1), padding='same')(tower_3)
#
#     output = concatenate([tower_1, tower_3], axis=3)
#     # output = Conv2D(32, (3, 4), padding='valid', activation='relu')(output)
#     # output = MaxPool2D((3, 1), strides=(1, 1), padding='valid')(output)
#     output = Conv2D(32, (5, 4), padding='valid', activation='relu')(output)
#     output = MaxPool2D((3, 1), strides=(1, 1), padding='valid')(output)
#     output = Flatten()(output)
#     output = Dense(32, activation='relu')(output)
#     out = Dense(5, activation='softmax')(output)
#     model = Model(inputs=input_img, outputs=out)
#
#     myoptimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
#     return model


def build_model(datasize=36):
    # datasize = DATASIZE
    W_maxnorm = 3
    DROPOUT = 0.5  #{{choice([0.3, 0.5, 0.7])}}
    model = Sequential()
    model.add(Conv2D(128, (6, 4), padding='same', input_shape=(datasize, 4, 1), activation='relu',
                     kernel_constraint=maxnorm(W_maxnorm)))
    # model.add(Conv2D(32, (3, 4), padding='same', input_shape=(datasize, 4, 1), activation='relu',
    #                  kernel_regularizer=regularizers.l2(0.01),
    #                  bias_regularizer=regularizers.l2(0.01),
    #                  activity_regularizer=regularizers.l1(0.01),
    #                  kernel_constraint=maxnorm(W_maxnorm)))
    model.add(BatchNormalization())
    # model.add(AveragePooling2D(pool_size=(3, 1), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(6, 4), strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (6, 4), padding='same', activation='relu',
                     kernel_constraint=maxnorm(W_maxnorm)))
    model.add(BatchNormalization())
    # model.add(AveragePooling2D(pool_size=(3, 1), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(6, 4), strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (6, 4), padding='same', activation='relu',
                     kernel_constraint=maxnorm(W_maxnorm)))
    model.add(BatchNormalization())
    # model.add(AveragePooling2D(pool_size=(3, 1), strides=(1, 1), padding='same'))
    # model.add(MaxPooling2D(pool_size=(3, 1), strides=(1, 1), padding='same'))
    # model.add(Conv2D(32, (3, 4),padding='same', activation='relu',
    #                  kernel_constraint=maxnorm(W_maxnorm)))

    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(3, 1), strides=(1, 1), padding='same'))
    # model.add(Conv2D(256, (5, 4),padding='same', activation='relu', kernel_constraint=maxnorm(W_maxnorm)))
    # model.add(MaxPool2D(pool_size=(5, 1), strides=(1, 1), padding='same'))
    # model.add(Conv2D(128, (5, 2),padding='same', activation='relu', kernel_constraint=maxnorm(W_maxnorm)))
    # model.add(MaxPool2D(pool_size=(5, 1), strides=(1, 1), padding='same'))
    # model.add(Conv2D(256, (5, 4),padding='same', activation='relu', kernel_constraint=maxnorm(W_maxnorm)))
    # model.add(MaxPool2D(pool_size=(5, 1), strides=(1, 1), padding='same'))

    model.add(Flatten())

    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    model.add(BatchNormalization())
    # model.add(activation('softmax'))
    adam1 = adam(lr=0.0001)  #, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # model.compile(loss='binary_crossentropy',
    #               optimizer=adam1,
    #               metrics=['binary_accuracy', 'fmeasure', 'precision', 'recall'])
    model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    return model


def train(model, X_train, Y_train):
    import time
    log_name = './Graph/' + str(time.time())
    tbCallBack = TensorBoard(log_dir=log_name, histogram_freq=1, write_graph=True, write_images=True, write_grads=True)
    history = model.fit(X_train, Y_train, batch_size=512, epochs=35, validation_split=0.2, shuffle=True, callbacks=[tbCallBack])
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
