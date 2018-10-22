from __future__ import print_function
import os
import numpy as np
import pickle
import random
import threading
import argparse

from data.load_data_label import load_label
from model import *
from data_generator import *

from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import backend as K

from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, Add, Flatten
from keras.models import Model
from keras.regularizers import l2

def hinge_loss(y_true,y_pred):
	# hinge loss
	y_pos = y_pred[:,:1]
	y_neg = y_pred[:,1:]
	loss = K.sum(K.maximum(0., args.margin - y_pos + y_neg))
	#print(loss.shape)
	return loss

def main(args):

	# load label
	train_list,valid_list,test_list,y_train_init,y_valid_init,y_test_init,artist_train,artist_valid,artist_test = load_label(args.num_sing,args.train_size,args.valid_size,args.test_size)
	print(len(train_list),len(valid_list),len(test_list),y_train_init.shape)
	y_valid = np.repeat(y_valid_init,args.num_segment,axis=0)

	# build_model
	if args.model == 'siamese':
		model = model_siamese(args.num_frame_input,args.N_negs)
	elif args.model == 'basic':
		model = model_basic(args.num_frame_input,args.num_sing)

	# model compile
	sgd = SGD(lr=args.lr,decay=args.lrdecay,momentum=0.9,nesterov=True)
	if args.model == 'siamese':
		model.compile(optimizer=sgd,loss=hinge_loss,metrics=['accuracy'])
	elif args.model == 'basic':
		model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()

	# load valid set
	if args.model == 'siamese':
		x_valid,y_valid = load_valid_siamese(valid_list,y_valid_init,artist_valid,args.mel_mean,
							args.mel_std,args.N_negs,args.feature_path,args.num_frame_input)
	elif args.model == 'basic':
		x_valid,y_valid = load_valid_basic(valid_list,y_valid_init,artist_valid,args.mel_mean,
							args.mel_std,args.num_sing,args.feature_path,args.num_frame_input)

	steps_per_epoch = int(len(train_list)/args.batch_size)
	weight_name = './models/model_%s_%d_%d_%.2f/weights.{epoch:02d}-{val_loss:.2f}.h5' % (args.model,args.num_sing,args.N_negs,args.margin)

	# make weight directory
	if not os.path.exists(os.path.dirname(weight_name)):
		os.makedirs(os.path.dirname(weight_name))

	#callbacks = [EarlyStopping(monitor='val_loss',patience=args.patience,verbose=1,mode='auto'),
	#				ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=1,verbose=1,mode='auto',min_lr=args.min_lr),
	#				ModelCheckpoint(monitor='val_loss',filepath=weight_name,verbose=0,save_best_only=True,mode='auto')]

	callbacks = [ModelCheckpoint(monitor='val_loss',filepath=weight_name,verbose=0,save_best_only=False,period=args.N,mode='auto')]

	# run model
	if args.model == 'siamese':
		model.fit_generator(generator=train_generator_siamese(train_list,y_train_init,artist_train,
							args.mel_mean,args.mel_std,args.N_negs,steps_per_epoch,args.batch_size,args.feature_path,args.num_frame_input),
					steps_per_epoch=steps_per_epoch,
					max_queue_size=5,
					workers=args.workers,
					use_multiprocessing=True,
					epochs=args.epochs,
					verbose=1,
					callbacks=callbacks,
					validation_data=(x_valid,y_valid))
	elif args.model == 'basic':
		model.fit_generator(generator=train_generator_basic(train_list,y_train_init,artist_train,
							args.mel_mean,args.mel_std,args.num_sing,steps_per_epoch,args.batch_size,args.feature_path,args.num_frame_input),
					steps_per_epoch=steps_per_epoch,
					max_queue_size=5,
					workers=args.workers,
					use_multiprocessing=True,
					epochs=args.epochs,
					verbose=1,
					callbacks=callbacks,
					validation_data=(x_valid,y_valid))

	print('training done!')


if __name__ == '__main__':

	# options
	parser = argparse.ArgumentParser(description='train artist siamese model')
	parser.add_argument('model', type=str, default='siamese', help='choose between siamese model and basic model')
	parser.add_argument('--train-size', type=int, default=15, help='the number of training samples out of 20 songs for each artist')
	parser.add_argument('--valid-size', type=int, default=18, help='the number of validation samples out of 20 songs for each artist')
	parser.add_argument('--test-size', type=int, default=20, help='the number of testing samples out of 20 songs for each artist')
	parser.add_argument('--num-sing', type=int, default=1000, help='the number of total artists used (500,1000,2000,5000,10000)')
	parser.add_argument('--feature-path', type=str, help='mel-spectrogram path')
	parser.add_argument('--N', type=int, default=10, help='save weight every N epochs')
	parser.add_argument('--min-lr', type=float, default=0.000016, help='minimum learning rate')
	parser.add_argument('--patience', type=int, default=6, help='learning rate reduce patience')
	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument('--lrdecay', type=float, default=1e-6, help='learning rate decaying')
	parser.add_argument('--epochs', type=int, default=1000, help='epochs')
	parser.add_argument('--batch-size', type=int, default=15, help='batch size')
	parser.add_argument('--workers', type=int, default=2, help='the number of generators to be used')
	parser.add_argument('--num-frame-input', type=int, default=129, help='frame size of input')
	parser.add_argument('--num-segment', type=int, default=10, help='the number of segments in a clip')
	parser.add_argument('--mel-bins', type=int, default=128, help='mel bin size')
	parser.add_argument('--N-negs', type=int, default=4, help='negative sampling size')
	parser.add_argument('--margin', type=float, default=0.4, help='margin value for hinge loss')
	parser.add_argument('--mel-mean', type=float, default=0.22620339, help='mean value calculated from training set')
	parser.add_argument('--mel-std', type=float, default=0.25794547, help='std value calculated from training set')
	args = parser.parse_args()

	main(args)







