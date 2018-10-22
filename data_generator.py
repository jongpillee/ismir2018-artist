from __future__ import print_function
import os
import numpy as np
import random
import threading
import argparse

class threadsafe_iter:
	def __init__(self,it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()

def threadsafe_generator(f):
	def g(*a, **kw):
		return threadsafe_iter(f(*a,**kw))
	return g

@threadsafe_generator
def train_generator_siamese(train_list,y_train,artist_train,mel_mean,mel_std,N_negs,steps_per_epoch,batch_size,feature_path,num_frame_input):
	# shuffling pos_anchor pos_item neg_items, audio 129 frames, random selection 
	while True:

		for batch_iter in range(0, steps_per_epoch*batch_size, batch_size):

			# initialization
			x_train_batch = []

			pos_anchor_x_train_batch = [] 
			pos_item_x_train_batch = []
			neg_items_x_train_batch = []


			for item_idx,item_iter in enumerate(range(batch_iter,batch_iter+batch_size)):

				# pos anchor
				file_path = feature_path + train_list[item_iter] + '.npy'			
				tmp = np.load(file_path)
				tmp = tmp.T

				tmp -= mel_mean
				tmp /= mel_std

				start = random.randint(0,tmp.shape[0]-num_frame_input)

				pos_anchor_x_train = tmp[start:start+num_frame_input,:]

				# pos item
				#print(artist_train,item_iter)
				pos_items = artist_train[y_train[item_iter]]
				pos_items_candidate = list(set(pos_items) - set([train_list[item_iter]]))
				pos_item = random.choice(pos_items_candidate)
				
				file_path = feature_path + pos_item + '.npy'
				tmp = np.load(file_path)
				tmp = tmp.T

				tmp -= mel_mean
				tmp /= mel_std

				start = random.randint(0,tmp.shape[0]-num_frame_input)

				pos_item_x_train = tmp[start:start+num_frame_input,:]

				# neg items
				neg_items_candidate = list(set(train_list) - set(pos_items))
				random.shuffle(neg_items_candidate)
				neg_items = neg_items_candidate[0:N_negs]
				
				neg_items_x_train = [[] for j in range(N_negs)]
				for neg_iter in range(N_negs):
					file_path = feature_path + neg_items[neg_iter] + '.npy'
					tmp = np.load(file_path)
					tmp = tmp.T

					tmp -= mel_mean
					tmp /= mel_std

					start = random.randint(0,tmp.shape[0]-num_frame_input)

					neg_items_x_train[neg_iter] = tmp[start:start+num_frame_input,:]

				pos_anchor_x_train_batch.append(pos_anchor_x_train) 
				pos_item_x_train_batch.append(pos_item_x_train)
				neg_items_x_train_batch.append(neg_items_x_train)

			pos_anchor_x_train_batch = np.array(pos_anchor_x_train_batch)
			pos_item_x_train_batch = np.array(pos_item_x_train_batch)
			neg_items_x_train_batch = np.array(neg_items_x_train_batch)

			x_train_batch = [pos_anchor_x_train_batch, pos_item_x_train_batch] + [neg_items_x_train_batch[:,j,:,:] for j in range(N_negs)]
		
			# y_train
			y_train_batch = np.zeros((batch_size,N_negs+1))
			y_train_batch[:,0] = 1
	
			yield x_train_batch, y_train_batch

	
def load_valid_siamese(valid_list,y_valid_init,artist_valid,mel_mean,mel_std,N_negs,feature_path,num_frame_input):
	# load valid sets
	pos_anchor_x_valid = [] 
	pos_item_x_valid = []
	neg_items_x_valid = []

	for item_iter in range(len(valid_list)):

		# pos anchor
		file_path = feature_path + valid_list[item_iter] + '.npy'			
		tmp = np.load(file_path)
		tmp = tmp.T

		tmp -= mel_mean
		tmp /= mel_std

		start = random.randint(0,tmp.shape[0]-num_frame_input)

		pos_anchor_x_tmp = tmp[start:start+num_frame_input,:]

		# pos item
		pos_items = artist_valid[y_valid_init[item_iter]]
		pos_items_candidate = list(set(pos_items) - set([valid_list[item_iter]]))
		pos_item = random.choice(pos_items_candidate)
				
		file_path = feature_path + pos_item + '.npy'
		tmp = np.load(file_path)
		tmp = tmp.T

		tmp -= mel_mean
		tmp /= mel_std

		start = random.randint(0,tmp.shape[0]-num_frame_input)

		pos_item_x_tmp = tmp[start:start+num_frame_input,:]

		# neg items
		neg_items_candidate = list(set(valid_list) - set(pos_items))
		random.shuffle(neg_items_candidate)
		neg_items = neg_items_candidate[0:N_negs]
				
		neg_items_x_tmp = [[] for j in range(N_negs)]
		for neg_iter in range(N_negs):
			file_path = feature_path + neg_items[neg_iter] + '.npy'
			tmp = np.load(file_path)
			tmp = tmp.T

			tmp -= mel_mean
			tmp /= mel_std

			start = random.randint(0,tmp.shape[0]-num_frame_input)

			neg_items_x_tmp[neg_iter] = tmp[start:start+num_frame_input,:]

		pos_anchor_x_valid.append(pos_anchor_x_tmp) 
		pos_item_x_valid.append(pos_item_x_tmp)
		neg_items_x_valid.append(neg_items_x_tmp)

		if np.remainder(item_iter,1000) == 0:
			print(item_iter)
	print(item_iter+1)
	
	pos_anchor_x_valid = np.array(pos_anchor_x_valid)
	pos_item_x_valid = np.array(pos_item_x_valid)
	neg_items_x_valid = np.array(neg_items_x_valid)

	x_valid = [pos_anchor_x_valid, pos_item_x_valid] + [neg_items_x_valid[:,j,:,:] for j in range(N_negs)]
		
	# y_train
	y_valid = np.zeros((len(valid_list),N_negs+1))
	y_valid[:,0] = 1

	return x_valid, y_valid


@threadsafe_generator
def train_generator_basic(train_list,y_train_init,artist_train,mel_mean,mel_std,num_sing,steps_per_epoch,batch_size,feature_path,num_frame_input):
	# shuffling pos_anchor pos_item neg_items, audio 129 frames, random selection
	while True:
		
		for batch_iter in range(0, steps_per_epoch*batch_size, batch_size):
			
			# initialization
			x_train_batch = []
			y_train_batch = np.zeros((batch_size,num_sing))

			for item_idx,item_iter in enumerate(range(batch_iter,batch_iter+batch_size)):

				# pos anchor
				file_path = feature_path + train_list[item_iter] + '.npy'
				tmp = np.load(file_path)
				tmp = tmp.T

				tmp -= mel_mean
				tmp /= mel_std

				start = random.randint(0,tmp.shape[0]-num_frame_input)

				pos_anchor_x_train = tmp[start:start+num_frame_input,:]

				x_train_batch.append(pos_anchor_x_train)

				y_train_batch[item_idx,int(y_train_init[item_iter])-1] = 1

			x_train_batch = np.array(x_train_batch)

			yield x_train_batch, y_train_batch

def load_valid_basic(valid_list,y_valid_init,artist_valid,mel_mean,mel_std,num_sing,feature_path,num_frame_input):
	
	x_valid = []
	y_valid = np.zeros((len(valid_list),num_sing))
	for item_iter in range(len(valid_list)):
		
		# pos anchor
		file_path = feature_path + valid_list[item_iter] + '.npy'
		tmp = np.load(file_path)
		tmp = tmp.T

		tmp -= mel_mean
		tmp /= mel_std

		start = random.randint(0,tmp.shape[0]-num_frame_input)

		pos_anchor_x_tmp = tmp[start:start+num_frame_input,:]
		x_valid.append(pos_anchor_x_tmp)

		y_valid[item_iter,int(y_valid_init[item_iter])-1] = 1

		if np.remainder(item_iter,1000) == 0:
			print(item_iter)
	print(item_iter)
	x_valid = np.array(x_valid)
	
	return x_valid, y_valid
	
