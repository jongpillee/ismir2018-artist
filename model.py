from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate
from keras.models import Model
from keras.regularizers import l2

def model_siamese(num_frame,J):
	pos_anchor = Input(shape = (num_frame,128))
	pos_item = Input(shape = (num_frame,128))
	neg_items = [Input(shape = (num_frame,128)) for j in range(J)]

	# item model **audio**
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
	
	# pos anchor
	pos_anchor_conv1 = conv1(pos_anchor)
	pos_anchor_bn1 = bn1(pos_anchor_conv1)
	pos_anchor_activ1 = activ1(pos_anchor_bn1)
	pos_anchor_MP1 = MP1(pos_anchor_activ1)
	pos_anchor_conv2 = conv2(pos_anchor_MP1)
	pos_anchor_bn2 = bn2(pos_anchor_conv2)
	pos_anchor_activ2 = activ2(pos_anchor_bn2)
	pos_anchor_MP2 = MP2(pos_anchor_activ2)
	pos_anchor_conv3 = conv3(pos_anchor_MP2)
	pos_anchor_bn3 = bn3(pos_anchor_conv3)
	pos_anchor_activ3 = activ3(pos_anchor_bn3)
	pos_anchor_MP3 = MP3(pos_anchor_activ3)
	pos_anchor_conv4 = conv4(pos_anchor_MP3)
	pos_anchor_bn4 = bn4(pos_anchor_conv4)
	pos_anchor_activ4 = activ4(pos_anchor_bn4)
	pos_anchor_MP4 = MP4(pos_anchor_activ4)
	pos_anchor_conv5 = conv5(pos_anchor_MP4)
	pos_anchor_bn5 = bn5(pos_anchor_conv5)
	pos_anchor_activ5 = activ5(pos_anchor_bn5)
	pos_anchor_sem = item_sem(pos_anchor_activ5)

	# pos item
	pos_item_conv1 = conv1(pos_item)
	pos_item_bn1 = bn1(pos_item_conv1)
	pos_item_activ1 = activ1(pos_item_bn1)
	pos_item_MP1 = MP1(pos_item_activ1)
	pos_item_conv2 = conv2(pos_item_MP1)
	pos_item_bn2 = bn2(pos_item_conv2)
	pos_item_activ2 = activ2(pos_item_bn2)
	pos_item_MP2 = MP2(pos_item_activ2)
	pos_item_conv3 = conv3(pos_item_MP2)
	pos_item_bn3 = bn3(pos_item_conv3)
	pos_item_activ3 = activ3(pos_item_bn3)
	pos_item_MP3 = MP3(pos_item_activ3)
	pos_item_conv4 = conv4(pos_item_MP3)
	pos_item_bn4 = bn4(pos_item_conv4)
	pos_item_activ4 = activ4(pos_item_bn4)
	pos_item_MP4 = MP4(pos_item_activ4)
	pos_item_conv5 = conv5(pos_item_MP4)
	pos_item_bn5 = bn5(pos_item_conv5)
	pos_item_activ5 = activ5(pos_item_bn5)
	pos_item_sem = item_sem(pos_item_activ5)

	# neg items
	neg_item_conv1s = [conv1(neg_item) for neg_item in neg_items]
	neg_item_bn1s = [bn1(neg_item_conv1) for neg_item_conv1 in neg_item_conv1s]
	neg_item_activ1s = [activ1(neg_item_bn1) for neg_item_bn1 in neg_item_bn1s]
	neg_item_MP1s = [MP1(neg_item_activ1) for neg_item_activ1 in neg_item_activ1s]
	neg_item_conv2s = [conv2(neg_item_MP1) for neg_item_MP1 in neg_item_MP1s]
	neg_item_bn2s = [bn2(neg_item_conv2) for neg_item_conv2 in neg_item_conv2s]
	neg_item_activ2s = [activ2(neg_item_bn2) for neg_item_bn2 in neg_item_bn2s]
	neg_item_MP2s = [MP2(neg_item_activ2) for neg_item_activ2 in neg_item_activ2s]
	neg_item_conv3s = [conv3(neg_item_MP2) for neg_item_MP2 in neg_item_MP2s]
	neg_item_bn3s = [bn3(neg_item_conv3) for neg_item_conv3 in neg_item_conv3s]
	neg_item_activ3s = [activ3(neg_item_bn3) for neg_item_bn3 in neg_item_bn3s]
	neg_item_MP3s = [MP3(neg_item_activ3) for neg_item_activ3 in neg_item_activ3s]
	neg_item_conv4s = [conv4(neg_item_MP3) for neg_item_MP3 in neg_item_MP3s]
	neg_item_bn4s = [bn4(neg_item_conv4) for neg_item_conv4 in neg_item_conv4s]
	neg_item_activ4s = [activ4(neg_item_bn4) for neg_item_bn4 in neg_item_bn4s]
	neg_item_MP4s = [MP4(neg_item_activ4) for neg_item_activ4 in neg_item_activ4s]
	neg_item_conv5s = [conv5(neg_item_MP4) for neg_item_MP4 in neg_item_MP4s]
	neg_item_bn5s = [bn5(neg_item_conv5) for neg_item_conv5 in neg_item_conv5s]
	neg_item_activ5s = [activ5(neg_item_bn5) for neg_item_bn5 in neg_item_bn5s]
	neg_item_sems = [item_sem(neg_item_activ5) for neg_item_activ5 in neg_item_activ5s]

	RQD_p = dot([pos_anchor_sem, pos_item_sem], axes = 1, normalize = True)
	RQD_ns = [dot([pos_anchor_sem, neg_item_sem], axes = 1, normalize = True) for neg_item_sem in neg_item_sems]

	prob = concatenate([RQD_p] + RQD_ns)

	# for hinge loss
	output = Activation('linear')(prob)

	model = Model(inputs = [pos_anchor, pos_item] + neg_items, outputs = output)
	return model

def model_basic(num_frame,num_sing):
	pos_anchor = Input(shape = (num_frame,128))

	# item model **audio**
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
	
	# pos anchor
	pos_anchor_conv1 = conv1(pos_anchor)
	pos_anchor_bn1 = bn1(pos_anchor_conv1)
	pos_anchor_activ1 = activ1(pos_anchor_bn1)
	pos_anchor_MP1 = MP1(pos_anchor_activ1)
	pos_anchor_conv2 = conv2(pos_anchor_MP1)
	pos_anchor_bn2 = bn2(pos_anchor_conv2)
	pos_anchor_activ2 = activ2(pos_anchor_bn2)
	pos_anchor_MP2 = MP2(pos_anchor_activ2)
	pos_anchor_conv3 = conv3(pos_anchor_MP2)
	pos_anchor_bn3 = bn3(pos_anchor_conv3)
	pos_anchor_activ3 = activ3(pos_anchor_bn3)
	pos_anchor_MP3 = MP3(pos_anchor_activ3)
	pos_anchor_conv4 = conv4(pos_anchor_MP3)
	pos_anchor_bn4 = bn4(pos_anchor_conv4)
	pos_anchor_activ4 = activ4(pos_anchor_bn4)
	pos_anchor_MP4 = MP4(pos_anchor_activ4)
	pos_anchor_conv5 = conv5(pos_anchor_MP4)
	pos_anchor_bn5 = bn5(pos_anchor_conv5)
	pos_anchor_activ5 = activ5(pos_anchor_bn5)
	pos_anchor_sem = item_sem(pos_anchor_activ5)

	output = Dense(num_sing, activation='softmax')(pos_anchor_sem)
	model = Model(inputs = pos_anchor, outputs = output)
	return model



	
