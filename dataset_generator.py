import pandas as pd
import sys
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from string import maketrans
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from scipy.stats.stats import pearsonr
from keras.regularizers import *


def oneHot(string):
	for x in range(len(string), 35):
		string = string+'A'
	trantab=maketrans('ACGT','0123')
	string=string+'ACGT'
#	data=list(string.translate({ord('A'):'0',ord('C'):'1',ord('G'):'2',ord('T'):'3'}))
	data=list(string.translate(trantab))
	return to_categorical(data)[0:-4]