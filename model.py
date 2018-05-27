from keras.models import Sequential
from keras.layers import Dense, Conv2D
import numpy


def build_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model