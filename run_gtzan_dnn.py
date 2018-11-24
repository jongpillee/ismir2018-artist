import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cPickle as cP

np.random.seed(0)

class Options(object):
	def __init__(self):
		self.train_size = 443#15244 #15244
		self.valid_size = 197#1529 #1529
		self.test_size = 290#4332 #4332
		self.num_tags = 10
		self.batch_size = 16
		self.nb_epoch = 1000
		self.lr = [0.01] #0.03 #0.06 # 0.01
		self.lrdecay = 1e-6 # e-6
		self.gpu_use = 1 # 1
		self.trial = 10
		self.num_neurons = [1024]
		self.dense_num_min = 0
		self.dense_num_max = 1
		self.activ = 'relu'
		self.regul = 'l2(1e-7)' # e-7
		self.init = 'he_uniform'
		self.optimizer = 'sgd' # adam, sgd # model 7 adam
		self.patience = 8
		self.mean = 0
		self.std = 0
		self.calculateNorm = 1

options = Options()

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

np.random.seed(0)  # for reproducibility                                              

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Convolution1D, MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.models import model_from_json, Model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

# imports 
import sys
from eval_tags import *

label_path = '/home1/irteam/users/jongpil/10.113.66.79:10101/data/GTZAN_split/'

# activation, regularizer, initialization
activ = options.activ
regul = eval(options.regul)
init = options.init

def build_model0(feature_length,num_neurons):

	pool_input = Input(shape=(feature_length,))

	output = Dense(options.num_tags,activation='softmax')(pool_input)
	model = Model(input=pool_input,output=output)
	
	return model

def build_model1(feature_length,num_neurons):

	pool_input = Input(shape=(feature_length,))

	dense1 = Dense(num_neurons,activation=activ)(pool_input)

	output = Dense(options.num_tags,activation='softmax')(dense1)
	model = Model(input=pool_input,output=output)
	
	return model

def build_model2(feature_length,num_neurons):

	pool_input = Input(shape=(feature_length,))

	dense1 = Dense(num_neurons,activation=activ)(pool_input)
	dense2 = Dense(num_neurons,activation=activ)(dense1)

	output = Dense(options.num_tags,activation='softmax')(dense2)
	model = Model(input=pool_input,output=output)
	
	return model

'''
class SGDLearningRateTracker(Callback):
	def on_epoch_end(self,epoch,logs={}):
		optimizer = self.model.optimizer

		# lr printer
		lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
		print('\nEpoch%d lr: %.6f' % (epoch+1,lr)) 
'''

def main(lr_iter,model_num,num_neurons,trial):

	model_name = 'model_siamese_hinge(num_frame_input,J)_5000'
	dataset = 'gtzan'
	J = 4
	margin = 0.4

	feature_path = '/home1/irteam/users/jongpil/10.113.66.79:10101/jp_artist/features/%s_%d_%.2f/%s/' % (model_name,J,margin,dataset)
	
	feature_length = 256  # 256
	
	# build model
	model_configuration = 'build_model%d(feature_length,num_neurons)' % model_num
	model = eval(model_configuration)

	# parameters
	batch_size = options.batch_size
	nb_epoch = options.nb_epoch
	lr = options.lr[lr_iter] #model1 0.01 model0 0.005 model2 0.005
	print 'lr: ' + str(lr)
	lrdecay = options.lrdecay

	# compile & optimizer
	sgd = SGD(lr=lr,decay=lrdecay,momentum=0.9,nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

	# print model summary
	print model.summary()

	# load path lists
	y = [[],[],[]]
	with open(label_path + 'train_filtered.txt') as f:
		train_list = f.read().splitlines()
		for line in train_list:
			y[0].append(line.split('/'))
	with open(label_path + 'valid_filtered.txt') as f:
		valid_list = f.read().splitlines()
		for line in valid_list:
			y[1].append(line.split('/'))
	with open(label_path + 'test_filtered.txt') as f:
		test_list = f.read().splitlines()
		for line in test_list:
			y[2].append(line.split('/'))
	
	y2 = [[],[],[]]
	y2[0] = np.zeros((len(y[0]),10))
	y2[1] = np.zeros((len(y[1]),10))
	y2[2] = np.zeros((len(y[2]),10))
	for iter in range(0,3):
		for iter2 in range(0,len(y[iter])):
			if y[iter][iter2][0] == 'blues':
				y2[iter][iter2,0] = 1
			elif y[iter][iter2][0] == 'classical':
				y2[iter][iter2,1] = 1
			elif y[iter][iter2][0] == 'country':
				y2[iter][iter2,2] = 1
			elif y[iter][iter2][0] == 'disco':
				y2[iter][iter2,3] = 1
			elif y[iter][iter2][0] == 'hiphop':
				y2[iter][iter2,4] = 1
			elif y[iter][iter2][0] == 'jazz':
				y2[iter][iter2,5] = 1
			elif y[iter][iter2][0] == 'metal':
				y2[iter][iter2,6] = 1
			elif y[iter][iter2][0] == 'pop':
				y2[iter][iter2,7] = 1
			elif y[iter][iter2][0] == 'reggae':
				y2[iter][iter2,8] = 1
			elif y[iter][iter2][0] == 'rock':
				y2[iter][iter2,9] = 1

	y_train = y2[0]
	y_valid = y2[1]
	y_test = y2[2]

	train_list = np.array(train_list)
	valid_list = np.array(valid_list)
	test_list = np.array(test_list)

	tmp1 = np.arange(len(train_list))
	np.random.shuffle(tmp1)
	train_list = train_list[tmp1]
	y_train = y_train[tmp1,:]



	# train size, valid size
	train_size = options.train_size #np.size(y_train,0)
	valid_size = options.valid_size #np.size(y_valid,0)
	test_size = options.test_size
	y_train = y_train[:train_size,:]
	y_valid = y_valid[:valid_size,:]
	y_test = y_test[:test_size,:]
	print y_train.shape, y_valid.shape, y_test.shape

	# load encoded feature
	x_train = np.zeros((train_size,feature_length))
	x_valid = np.zeros((valid_size,feature_length))

	for iter in range(0,train_size):
		file_path = feature_path + train_list[iter] + '.npy'
		x_train[iter] = np.load(file_path)

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1

	for iter in range(0,valid_size):
		file_path = feature_path + valid_list[iter] + '.npy'
		x_valid[iter] = np.load(file_path)

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1

	print train_size,valid_size
	print x_train.shape, x_valid.shape

	if options.calculateNorm == 1:
		mean_value = np.mean(x_train)
		std_value = np.std(x_train)
	else:
		mean_value =1 
		std_value = 1

	x_train -= mean_value
	x_valid -= mean_value
	x_train /= std_value
	x_valid /= std_value

	print 'mean value: ' + str(mean_value)
	print 'std value: ' + str(std_value)
	print 'Normalization done!'

	options.mean = mean_value
	options.std = std_value

	indicator = '[neurons:%d][model_num:%d][lr:%f][trial:%d]' % (num_neurons,model_num,lr,trial)

	# Callbacks
	weight_name = './weights/%s_%d_%.2f/%s/best_weights_dnn_%s.hdf5' % (model_name,J,margin,dataset,indicator)

	if not os.path.exists(os.path.dirname(weight_name)):
		os.makedirs(os.path.dirname(weight_name))

	checkpointer = ModelCheckpoint(weight_name,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
	earlyStopping = EarlyStopping(monitor='val_loss',patience=options.patience,verbose=0,mode='auto')
	# train start
	hist = model.fit(x_train,y_train,callbacks=[earlyStopping,checkpointer],batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,
						validation_data=(x_valid,y_valid))

	# save model architecture
	json_string = model.to_json()
	json_name = './weights/%s_%d_%.2f/%s/model_architecture_dnn_%s.json' % (model_name,J,margin,dataset,indicator)
	open(json_name,'w').write(json_string)

	# load best model weights for testing
	model.load_weights(weight_name)
	print 'best model loaded for testing'

	#------------- test set!! ----------------------------------------------------------------
	del x_train

	x_test = np.zeros((test_size,feature_length))
	for iter in range(0,test_size):
		file_path = feature_path + test_list[iter] + '.npy'
		x_test[iter] = np.load(file_path)

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1

	x_test -= mean_value
	x_test /= std_value

	output = model.predict(x_test)

	print output.shape
	print y_test.shape

	acc = eval_tops(output,y_test,1)

	print 'model_num_of_dense_layer: ' + str(model_num)
	print 'num_neurons: ' + str(num_neurons)
	print ('acc: %.4f' % (acc))

	save_dir = '/home1/irteam/users/jongpil/10.113.66.79:10101/jp_artist/results/%s_%d_%.2f/%s/%s_%.3f' % (model_name,J,margin,dataset,indicator,acc)
	
	if not os.path.exists(os.path.dirname(save_dir)):
		os.makedirs(os.path.dirname(save_dir))

	save_list = []

	cP.dump(save_list,open(save_dir,'w'))
	print 'result save done!!!'
        
        return acc

if __name__ == '__main__':

	num_neurons = options.num_neurons

	for lr_iter in range(0,len(options.lr)):
		for model_num in range(options.dense_num_min,options.dense_num_max):
			for iter2 in range(len(num_neurons)):
                                acc_list = []
				for trial_iter in range(options.trial):
					accuracy = main(lr_iter,model_num,num_neurons[iter2],trial_iter)
                                        acc_list.append(accuracy)

                                print acc_list
                                print np.mean(acc_list)
                               










