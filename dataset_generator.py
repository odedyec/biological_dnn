from keras.utils import to_categorical
import numpy as np
import pandas as pd
import time


def label_generator(size1, size2):
	"""
	Create a two coloum label set
	:param size1:
	:param size2:
	:return:
	"""
	lab1 = np.concatenate((
                        np.ones((size1, 1), dtype=float),
                        np.zeros((size1, 1), dtype=float)),
                   axis=1)
	lab2 = np.concatenate((
        np.zeros((size2, 1), dtype=float),
        np.ones((size2, 1), dtype=float)),
        axis=1)

	lab = np.concatenate((lab1, lab2), axis=0)
	return lab


def suffle_data_label(a, b):
	c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
	np.random.shuffle(c)
	a2 = c[:, :a.size // len(a)].reshape(a.shape)
	b2 = c[:, a.size // len(a):].reshape(b.shape)
	return a2, b2


def split_train_test(selex0, selex1, train_size):
	"""
	split the selex inputs into train and test sets
	:param selex0:
	:param selex1:
	:param train_size:
	:return:
	"""
	np.random.shuffle(selex0)
	np.random.shuffle(selex1)
	x_train = np.concatenate((selex0[0:int(train_size/2), :, :, :], selex1[0:int(train_size/2), :, :, :]), axis=0)
	x_test = np.concatenate((selex0[int(train_size / 2)+1:, :, :, :], selex1[int(train_size / 2)+1:, :, :, :]), axis=0)

	y_train = label_generator(int(train_size/2), int(train_size/2))

	y_test = label_generator(int(len(selex0)-train_size/2-1), int(len(selex1)-train_size/2-1))

	return x_train, x_test, y_train, y_test



def oneHot(string):
	"""
	One Hot encoding of a DNA sequence ACGT -> 0123 -> [1, 0, 0, 0], [0, 1, 0, 0], ...
	:param string: DNA sequence with A, C, G, or T
	:return: a matrix of the encoded sequence
	"""

	string = string.replace('A', '0')
	string = string.replace('C', '1')
	string = string.replace('G', '2')
	string = string.replace('T', '3')
	data = list(map(int, string))
	return to_categorical(data, 4)


def pbm_dataset_generator(filename):
	"""
	Open a PBM file andtransform it to a numpy encoded list
	:param filename: a PBM file with N lines of DNA sequences
	:return: an Nx4 numpy array
	"""
	f = open(filename, 'r')
	data = []
	for line in f:
		data.append(oneHot(line[0:60]))
	f.close()
	arr = np.array(data)
	return arr


def selex_dataset_generator(filename):
	"""
	Open a selex file and transform it to two numpy lists.
	One for the onehot encoded DNA sequence
	One for the number of occurences
	:param filename:
	:return:
	"""

	dat = pd.read_csv(filename, delimiter='\t', usecols=[0], header=-1)
	f = dat.get(0)
	data = []

	t = time.time()
	for line2 in f:
		encoded_line = oneHot(line2)
		# if (encoded_line.shape != (20, 4)):
		# 	print "Warning! not a (20, 4) shape, but", encoded_line.shape
		# 	continue
		# print encoded_line.shape
		padding = int((36 - len(encoded_line)) / 2)
		encoded_line = np.concatenate((np.concatenate((0.25 * np.ones((padding, 4)), encoded_line)), 0.25 * np.ones((padding, 4))))
		data.append(encoded_line)

	data = np.asarray(data)
	print('Took ', time.time() - t, ' seconds to encode data, for ', filename)
	return data, None


def save_dataset(x_train, x_test, y_train, y_test):
	import h5py
	hdf5_file = h5py.File('data_tf1.hdf5', mode='w')
	hdf5_file.create_dataset("x_train", data = x_train)
	hdf5_file.create_dataset("y_train", data = y_train)
	hdf5_file.create_dataset("x_test", data=x_test)
	hdf5_file.create_dataset("y_test", data=y_test)
	hdf5_file.close()

def load_dataset():
	import h5py
	f = h5py.File('data_tf1.hdf5', 'r')
	x_train = f["x_train"].value
	y_train = f["y_train"].value
	x_test = f["x_test"].value
	y_test = f["y_test"].value
	return x_train, x_test, y_train, y_test


def generate_data(PBM_FILE, SELEX_FILES, GENERATE_DATASET=True, TRAIN_SIZE=100000, SELEX_SIZE=36):
	pbm_data = pbm_dataset_generator(PBM_FILE)
	if GENERATE_DATASET:  # load data and OneHot encode data
		print(pbm_data.shape)
		selex_4, _ = selex_dataset_generator(SELEX_FILES[-1])
		selex_0, _ = selex_dataset_generator(SELEX_FILES[0])

		selex_4 = selex_4.reshape((len(selex_4), SELEX_SIZE, 4, 1))
		selex_0 = selex_0.reshape((len(selex_0), SELEX_SIZE, 4, 1))

		x_train, x_test, y_train, y_test = split_train_test(selex_0, selex_4, TRAIN_SIZE)
		save_dataset(x_train, x_test, y_train, y_test)
	else:  # Load from data_tf1.hdf5 file
		t = time.time()
		x_train, x_test, y_train, y_test = load_dataset()
		print('Took ', time.time() - t, ' seconds to load data from h5py file')
	return x_train, x_test, y_train, y_test, pbm_data

import sys

def get_argv():
    """
    Get input from sys.argv
    :return:
    """
    if len(sys.argv) < 3:
        print("Length of input arguments is ", len(sys.argv))
        print("\nUsage:\n python main.py pbm_file SELEX_FILE_0 SELEX_FILE_1 ...")
        print("\nUsage2:\n python main.py pbm_file #of_selex_0 #of_selex_1 ...")
        sys.exit(0)

    PBM_FILE = sys.argv[1]
    SELEX_FILES = [sys.argv[i] for i in range(2, len(sys.argv))]
    if SELEX_FILES[0].isdigit():
        SELEX_FILES = list(map(int, SELEX_FILES))
    return PBM_FILE, SELEX_FILES


def parse_args(PBM_FILE, SELEX_FILES):
    """
    Transform selex numbers to filenames.
    :param PBM_FILE: Required for full path
    :param SELEX_FILES: List of numbers of SELEX cycles, or filenames
    :return: Filenames of everything
    """
    if len(SELEX_FILES) < 1:
        parse_args()
    if type(SELEX_FILES[0]) == int:
        base = PBM_FILE.split('_')[0]
        selex = [base+'_selex_'+str(i)+'.txt' for i in SELEX_FILES]
    return PBM_FILE, selex


if __name__ == '__main__':
	PBM_FILE, SELEX_FILES = 'train/TF1_pbm.txt', [0, 1, 2, 3, 4]  # get_argv()
	print(PBM_FILE)
	print(SELEX_FILES)
	# PBM_FILE, SELEX_FILES =
	PBM_FILE, SELEX_FILES = parse_args(PBM_FILE, SELEX_FILES)
	generate_data(PBM_FILE, SELEX_FILES)


