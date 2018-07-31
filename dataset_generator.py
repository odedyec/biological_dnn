from keras.utils import to_categorical
import numpy as np
import pandas as pd
import time

TRAIN_SIZE = 30000
TEST_SIZE = 30000

def label_generator(num_of_labels, size):
	"""
	Create a two coloum label set
	:param size1:
	:param size2:
	:return:
	"""
	lab = np.zeros((size, num_of_labels), dtype=np.float)
	length_per_label = int(size / num_of_labels)
	for i in range(num_of_labels):
		lab[(i * length_per_label):((i+1) * length_per_label), i] = 1.0

	return lab


def suffle_data_label(a, b):
	c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
	np.random.shuffle(c)
	a2 = c[:, :a.size // len(a)].reshape(a.shape)
	b2 = c[:, a.size // len(a):].reshape(b.shape)
	return a2, b2


def split_train_test(selex_data, train_size, test_size):
	"""
	split the selex inputs into train and test sets
	:param selex0:
	:param selex1:
	:param train_size:
	:return:
	"""
	split_size = len(selex_data)
	x_train = np.array([])
	x_test = np.array([])
	for selex in selex_data:
		np.random.shuffle(selex)
		if len(x_test) == 0:
			x_train = selex[0:int(train_size/split_size), :, :, :]
			x_test = selex[int(train_size / split_size)+1:int(train_size / split_size)+1+int(test_size/split_size), :, :, :]
			continue
		x_train = np.concatenate((x_train, selex[0:int(train_size/split_size), :, :, :]), axis=0)
		x_test = np.concatenate((x_test, selex[int(train_size / split_size)+1:int(train_size / split_size)+1+int(test_size/split_size), :, :, :]), axis=0)

	y_train = label_generator(split_size, train_size)
	y_test = label_generator(split_size, test_size)

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


def selex_dataset_generator(filename, data_to_load=TRAIN_SIZE+TEST_SIZE, selex_size=36):
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
	k = 0;
	t = time.time()
	for line2 in f:
		if k == data_to_load:
			break
		k += 1
		encoded_line = oneHot(line2)
		# if (encoded_line.shape != (20, 4)):
		# 	print "Warning! not a (20, 4) shape, but", encoded_line.shape
		# 	continue
		# print encoded_line.shape
		padding = int((selex_size - len(encoded_line)) / 2)
		encoded_line = np.concatenate((np.concatenate((0.25 * np.ones((padding, 4)), encoded_line)), 0.25 * np.ones((padding, 4))))
		data.append(encoded_line)
		encoded_line_rev = np.zeros(encoded_line.shape)
		encoded_line_rev[:, 0] = encoded_line[:, 2]
		encoded_line_rev[:, 1] = encoded_line[:, 3]
		encoded_line_rev[:, 2] = encoded_line[:, 0]
		encoded_line_rev[:, 3] = encoded_line[:, 1]
		encoded_line_rev = encoded_line_rev[::-1]
		data.append(encoded_line_rev)

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


def generate_data(PBM_FILE, SELEX_FILES, GENERATE_DATASET=True, train_size=100000, SELEX_SIZE=36, test_size=100000):
	pbm_data = pbm_dataset_generator(PBM_FILE)
	if GENERATE_DATASET:  # load data and OneHot encode data
		print(pbm_data.shape)
		selex_4, _ = selex_dataset_generator(SELEX_FILES[-1], (train_size+test_size)/5+1, selex_size=SELEX_SIZE)
		selex_3, _ = selex_dataset_generator(SELEX_FILES[-2], (train_size+test_size)/5+1, selex_size=SELEX_SIZE)
		selex_2, _ = selex_dataset_generator(SELEX_FILES[-3], (train_size+test_size)/5+1, selex_size=SELEX_SIZE)
		selex_1, _ = selex_dataset_generator(SELEX_FILES[-4], (train_size+test_size)/5+1, selex_size=SELEX_SIZE)
		selex_0, _ = selex_dataset_generator(SELEX_FILES[0], (train_size+test_size)/5+1, selex_size=SELEX_SIZE)

		selex_data = list()
		selex_data.append(selex_0.reshape((len(selex_0), SELEX_SIZE, 4, 1)))
		selex_data.append(selex_1.reshape((len(selex_1), SELEX_SIZE, 4, 1)))
		selex_data.append(selex_2.reshape((len(selex_2), SELEX_SIZE, 4, 1)))
		selex_data.append(selex_3.reshape((len(selex_3), SELEX_SIZE, 4, 1)))
		selex_data.append(selex_4.reshape((len(selex_4), SELEX_SIZE, 4, 1)))

		x_train, x_test, y_train, y_test = split_train_test(selex_data, 2*train_size, 2*test_size)
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
	generate_data(PBM_FILE, SELEX_FILES, test_size=TEST_SIZE, train_size=TRAIN_SIZE)


