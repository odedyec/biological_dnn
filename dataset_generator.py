from keras.utils import to_categorical
from string import maketrans
import numpy as np



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
	# for x in range(len(string)):
		# string = string+'A'
	trantab=maketrans('ACGT','0123')
	# string=string+'ACGT'
#	data=list(string.translate({ord('A'):'0',ord('C'):'1',ord('G'):'2',ord('T'):'3'}))
	data=list(string.translate(trantab))
	data = map(int, data)
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
	return np.array(data)


def selex_dataset_generator(filename):
	"""
	Open a selex file and transform it to two numpy lists.
	One for the onehot encoded DNA sequence
	One for the number of occurences
	:param filename:
	:return:
	"""
	f = open(filename, 'r')
	data = []
	labels = []
	for line in f:
		line2 = line.split('\t')
		encoded_line = oneHot(line2[0])
		if (encoded_line.shape != (20, 4)):
			print "Warning! not a (20, 4) shape, but", encoded_line.shape
			continue
		# print encoded_line.shape
		data.append(encoded_line)
		labels.append(int(line2[1]))
	f.close()
	data = np.asarray(data)
	labels = np.array(labels)
	return data, labels
