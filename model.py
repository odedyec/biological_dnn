from keras.models import Sequential
from keras.layers import Dense, Conv2D
import pandas as pd
import sys
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from scipy.stats.stats import pearsonr
from keras.regularizers import *
import numpy as np


def build_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model