from __future__ import print_function
import os
import numpy as np
import time
import cPickle as cP
import argparse

from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2

from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate
from keras.models import Model

parser = argparse.ArgumentParser(description='encoding feature')
parser.add_argument('model', type=str, help='siamese or basic') 
parser.add_argument('weightpath', type=str, help='weight path')
parser.add_argument('--num-sing', type=int, default=2000, help='the number of artists used for basic model')
args = parser.parse_args()

num_frames_per_song = 1291
img_cols = 128
num_frame_input = 129
num_segment = int(num_frames_per_song/num_frame_input)
print('Number of segments per song: ' + str(num_segment))

audiolist = 'List of audio files. ex) train_filtered.txt, blues/blues.00029.wav, ... line by line'
audio_path = 'AUDIO PATH'
save_path = './features/%s/' % args.model

# load data
with open(audiolist) as f:
	all_list = f.read().splitlines()
print(len(all_list))

# path generate
if not os.path.exists(os.path.dirname(save_path)):
	os.makedirs(os.path.dirname(save_path))

def load_melspec(file_name_from,num_segment,num_frame_input):
	file_name = audio_path + file_name_from.replace('.wav','.npy')
	tmp = np.load(file_name)
	tmp = tmp.T

	mel_feat = np.zeros((num_segment,num_frame_input,128))
	for iter2 in range(0,num_segment):
		mel_feat[iter2] = tmp[iter2*num_frame_input:(iter2+1)*num_frame_input,:]
	
	return mel_feat


# load model
model_input = Input(shape = (num_frame_input,128))

conv1 = Conv1D(128,4,padding='same',use_bias=True,kernel_regularizer=l2(1e-5),kernel_initializer='he_uniform')
bn1 = BatchNormalization()
activ1 = Activation('relu')
MP1 = MaxPool1D(pool_size=4)
conv2 = Conv1D(128,4,padding='same',use_bias=True,kernel_regularizer=l2(1e-5),kernel_initializer='he_uniform')
bn2 = BatchNormalization()
activ2 = Activation('relu')
MP2 = MaxPool1D(pool_size=4)
conv3 = Conv1D(128,4,padding='same',use_bias=True,kernel_regularizer=l2(1e-5),kernel_initializer='he_uniform')
bn3 = BatchNormalization()
activ3 = Activation('relu')
MP3 = MaxPool1D(pool_size=4)
conv4 = Conv1D(128,2,padding='same',use_bias=True,kernel_regularizer=l2(1e-5),kernel_initializer='he_uniform')
bn4 = BatchNormalization()
activ4 = Activation('relu')
MP4 = MaxPool1D(pool_size=2)
conv5 = Conv1D(256,1,padding='same',use_bias=True,kernel_regularizer=l2(1e-5),kernel_initializer='he_uniform')
bn5 = BatchNormalization()
activ5 = Activation('relu')
drop1 = Dropout(0.5)

item_sem = GlobalAvgPool1D()

model_conv1 = conv1(model_input)
model_bn1 = bn1(model_conv1)
model_activ1 = activ1(model_bn1)
model_MP1 = MP1(model_activ1)
model_conv2 = conv2(model_MP1)
model_bn2 = bn2(model_conv2)
model_activ2 = activ2(model_bn2)
model_MP2 = MP2(model_activ2)
model_conv3 = conv3(model_MP2)
model_bn3 = bn3(model_conv3)
model_activ3 = activ3(model_bn3)
model_MP3 = MP3(model_activ3)
model_conv4 = conv4(model_MP3)
model_bn4 = bn4(model_conv4)
model_activ4 = activ4(model_bn4)
model_MP4 = MP4(model_activ4)
model_conv5 = conv5(model_MP4)
model_bn5 = bn5(model_conv5)
model_activ5 = activ5(model_bn5)
model_drop1 = drop1(model_activ5)

model_item_sem = item_sem(model_drop1)

if args.model == 'siamese':
	RQD_p = dot([model_item_sem, model_item_sem], axes = 1, normalize = True)
	output = Activation('linear')(RQD_p)
elif args.model == 'basic':
	output = Dense(args.num_sing,activation='softmax')(model_item_sem)

model = Model(inputs = model_input, outputs = output)

model.load_weights(args.weightpath)
print('model loaded!!!')


# compile & optimizer
sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

# print model summary
model.summary()

# mean / std
mel_mean = 0.22620339
mel_std = 0.25794547

# define activation layer
layer_dict = dict([(layer.name,layer) for layer in model.layers[1:]])
activation_layer='activation_5'
layer_output = layer_dict[activation_layer].output
get_last_hidden_output = K.function([model.layers[0].input, K.learning_phase()], [layer_output])

# encoding
all_size = len(all_list)
for iter2 in range(0,len(all_list)):
	# check existence
	save_name = save_path + all_list[iter2].replace('.wav','.npy')
	
	if not os.path.exists(os.path.dirname(save_name)):
		os.makedirs(os.path.dirname(save_name))
	
	if os.path.isfile(save_name) == 1:
		print(iter2, save_name + '_file_exist')

	# load melgram
	x_mel_tmp = load_melspec(all_list[iter2],num_segment,num_frame_input)

	# normalization
	x_mel_tmp -= mel_mean
	x_mel_tmp /= mel_std

	# prediction
	weight = get_last_hidden_output([x_mel_tmp,0])[0] # testing phase 0
	print(weight.shape) # 10,1,256

	maxpooled = np.amax(weight,axis=1)
	averagepooled = np.average(maxpooled,axis=0)
	print(averagepooled.shape,iter2)
	
	np.save(save_name,averagepooled)


