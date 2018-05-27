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

epo=10
mink=5
#maxk=5
maxk=6
mini=16
#maxi=2
maxi=17
def oneHot(string):
	for x in range(len(string), 35):
		string = string+'A'
	trantab=maketrans('ACGT','0123')
	string=string+'ACGT'
#	data=list(string.translate({ord('A'):'0',ord('C'):'1',ord('G'):'2',ord('T'):'3'}))
	data=list(string.translate(trantab))
	return to_categorical(data)[0:-4]

if len(sys.argv) > 2:
	file=sys.argv[1] 
	file2=sys.argv[2]
elif len(sys.argv) > 1:
	file='HK/pTH'+sys.argv[1]+'_HK.raw.out'
	file2='ME/pTH'+sys.argv[1]+'_ME.raw.out'
else:
	file='HK/pTH1049_HK.raw.out'
	file2='ME/pTH1049_ME.raw.out'

pbm=pd.read_csv(file, sep='\t', header=None, names=['seq','int'])
data=map(oneHot, pbm['seq'])
#data=np.array
#for seq in pbm['seq']:
#	np.append(data, oneHot(seq))
values=pbm['int']

max = 0
mi = 0
mk = 0
v = 0.67
# hyper-parameter testing
for i in range(mini,maxi):
	for k in range(mink,maxk):
		model=Sequential()
		model.add(Conv1D(filters=i, kernel_size=k, strides=1, kernel_initializer='RandomNormal', activation='relu', 
		kernel_regularizer=l2(5e-3), input_shape=(35,4), use_bias=True, bias_initializer='RandomNormal'))
#model.add(Dropout(0.2))
#		model.add(MaxPooling1D(pool_size=1))
#model.add(Dense(1, input_shape=(35,4)))
#model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense((35-k+1)*i, activation='relu'))
#		model.add(Dense(128))
#		model.add(Dense(128))
#		model.add(Dropout(0.5))
		model.add(Dense(1)) #, activation='relu'))

		sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd, loss='mse')
#              loss='kullback_leibler_divergence')

		l=(int)(len(data)*v)
		model.fit(np.array(data[0:l]), np.array(values[0:l]), epochs=epo, batch_size=16, verbose=1)
		model.layers[0].get_weights()
#model.fit(np.array(data), np.random.uniform(0,1,40330), epochs=10, batch_size=32)

		y2=model.predict(np.array(data[l+1:len(data)]))
		p=pearsonr(y2.reshape(len(values[l+1:len(data)])),np.array(values[l+1:len(data)]))[0]
		print(i, k, p)
		if (p>max):
			max=p
			mi = i
			mk = k
print(mi, mk, max)

for i in range(mi,mi+1):
	for k in range(mk,mk+1):
		model=Sequential()
		model.add(Conv1D(filters=i, kernel_size=k, strides=1, kernel_initializer='glorot_normal', activation='relu',
		kernel_regularizer=l2(5e-6), input_shape=(35,4), use_bias=True))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(1, activation='relu'))

		sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=False)
		model.compile(optimizer=sgd, loss='mse')
		model.fit(np.array(data), np.array(values), epochs=epo, batch_size=16, verbose=1)

		pbm2=pd.read_csv(file2, sep='\t', header=None, names=['seq','int'])
		data2=map(oneHot, pbm2['seq'])
		values2=pbm2['int']
		y2=model.predict(np.array(data2))
		p=pearsonr(y2.reshape(len(values2)),np.array(values2))[0]
		print(i, k, p)
