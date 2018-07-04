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
import numpy as np

def oneHot(string):
	# for x in range(len(string)):
		# string = string+'A'
	trantab=maketrans('ACGT','0123')
	# string=string+'ACGT'
#	data=list(string.translate({ord('A'):'0',ord('C'):'1',ord('G'):'2',ord('T'):'3'}))
	data=list(string.translate(trantab))
	data = map(int, data[0:60])
	return to_categorical(data)


def pbm_dataset_generator(filename):
	f = open(filename, 'r')
	data = []
	for line in f:
		data.append(oneHot(line))
	f.close()
	return data

def selex_dataset_generator(filename):
	f = open(filename, 'r')
	data = []
	labels = []
	for line in f:
		line2 = line.split('\t')
		data.append(oneHot(line2[0]))
		labels.append(int(line2[1]))
	f.close()
	return data, labels
