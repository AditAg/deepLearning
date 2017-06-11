from keras import backend as K
from keras.layers.convolutional import Conv1D,ZeroPadding1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense,Flatten,Activation,Dropout,Reshape
from keras.layers import Input,concatenate
from keras.models import Model,Sequential    
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import tensorflow as tf
import math
import theano

#m = 36864					       # total no. of training examples(i.e. no of P(plaintext)-K(key) pairs)
m = 12288                                             # we have total 20000 batches each of size 4096 each and at 1 time we generate 3 batches -1 for AliceBob and 2 for Eve
# can optimize the mini_batch_size to improve performance   (ranges from 256 to 4096)
mini_batch_size = 4096     # mini-batch size for calculating estimated values over the minibatches instead of expected values over a distribution.
# can change below for more robustness
plaintext_size = 16				       # no. of bits in plaintext
key_size = 16	                                       # no. of bits in Key
no_input_neurons = plaintext_size + key_size
ciphertext_size = plaintext_size
NO_EPOCHS = 5
learning_rate = 0.0008

i=0

def get_data(size1,size2):
	# get the proper data here,currently generating m random message+key pairs
	return np.random.randint(0,2,size = (size1,size2))*2 -1
train_data = get_data(m,no_input_neurons)

def L1_distance(x1,x2):                                # can optimize here by considering different distance functions as compared to L1 norm
	assert (x1.shape==x2.shape)
	sum = 0
	for i in range(x1.shape[0]):
		L1_val = abs(x1[i] - x2[i])
		sum+=L1_val
	return sum

def expected_value(x):                      # x is a matrix of size (no. of examples considered(should be = minibatchsize))X(size of each example)
	sum=np.zeros((x.shape[1]))
	for i in range(x.shape[0]):
		sum+= x[i]
	sum = (sum*1.0)/m                              # considering probability of each training example to be 1/m
	return sum

#def loss_function(ciphertext):
#	def custom_objective(y_true,y_pred):
#		global i
#		reconstruction_error = tf.scalar_mul(1.0/m,(tf.reduce_sum(tf.abs(y_true - y_pred),1)))
#		sess = tf.Session()
#		sess.run(ciphertext)
#		eavesdropper_success = tf.scalar_mul(1.0/m,tf.reduce_sum(tf.abs(y_true-Eve_model.predict(ciphertext.eval(sess))),1))
#		tt = tf.constant(value = plaintext_size/2,shape = eavesdropper_success.get_shape())	
#		eavesdropper_success = tf.square(tt - eavesdropper_success)
#		eavesdropper_success = tf.scalar_mul((4.0/(plaintext_size**2)),(eavesdropper_success))
#		return K.mean(reconstruction_error + eavesdropper_success)
#	return custom_objective

def custom_objective(y_true,y_pred):
	global i
	reconstruction_error = tf.scalar_mul(1.0/m,(tf.reduce_sum(tf.abs(y_true - y_pred),1)))
	data = train_data[i:i+mini_batch_size]
	aux_data = data[:,ciphertext_size:no_input_neurons]
	intermediate_output = intermediate_layer_model.predict([data,aux_data])
	eavesdropper_success = tf.scalar_mul(1.0/m,tf.reduce_sum(tf.abs(y_true-Eve_model.predict(intermediate_output)),1))
	tt = tf.constant(value = plaintext_size/2,shape = eavesdropper_success.get_shape())	
	eavesdropper_success = tf.square(tt - eavesdropper_success)
	eavesdropper_success = tf.scalar_mul((4.0/(plaintext_size**2)),(eavesdropper_success))
	return K.mean(reconstruction_error + eavesdropper_success)

def custom_objective2(y_true,y_pred):
	return K.mean(tf.scalar_mul(1.0/m,(tf.reduce_sum(tf.abs(y_true - y_pred),1))))

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
	
# Alice's network
# FC layer -> Conv Layer (4 1-D convolutions)	
input_layer = Input(shape=(no_input_neurons,))
l1 = Dense(no_input_neurons, activation = 'sigmoid',kernel_initializer = 'glorot_normal')(input_layer)
l2 = Reshape((no_input_neurons,1))(l1)
l3 = ZeroPadding1D(padding=2)(l2)
# Xavier Glotrot Initialization of weights
l4 = Conv1D(2,kernel_size = 4, strides = 1, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l3)
l5 = Conv1D(4,kernel_size = 2, strides = 2, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l4)
l6 = Conv1D(4,kernel_size = 1, strides = 1, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l5)
l7 = Conv1D(1,kernel_size = 1, strides = 1, activation = 'tanh', kernel_initializer = 'glorot_normal')(l6)
l8 = Flatten(name = 'cipher_output')(l7)

# For Bob's Network
# FC layer -> Conv Layer (4 1-D convolutions)
auxillary_input = Input(shape = (key_size,),name = 'aux_input')
new_input = concatenate([l8,auxillary_input],axis=1)
x = Dense(no_input_neurons, activation = 'sigmoid',kernel_initializer = 'glorot_normal')(new_input)
x1 = Reshape((no_input_neurons,1))(x)
x2 =ZeroPadding1D(padding =2)(x1)
x3 = Conv1D(2,kernel_size = 4, strides = 1, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(x2)
x4 = Conv1D(4,kernel_size = 2, strides = 2, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(x3)
x5 = Conv1D(4,kernel_size = 1, strides = 1, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(x4)
x6 = Conv1D(1,kernel_size = 1, strides = 1, activation = 'tanh', kernel_initializer = 'glorot_normal')(x5)
Bob_output = Flatten()(x6)

#For Eve
input_eve = Input(shape = (ciphertext_size,))                        # Imp: When we train Eve then it should only modify his own parameters and not Alice's networks parameters since it is "optimal Eve" is : argmin over theta_eve (loss of Eve) so l9 is not applied on l8
l9 = Dense(no_input_neurons, activation = 'sigmoid',kernel_initializer = 'glorot_normal')(input_eve)
l10 = Dense(no_input_neurons, activation = 'sigmoid',kernel_initializer = 'glorot_normal')(l9)
l11 = Reshape((no_input_neurons,1))(l10)
l12 = ZeroPadding1D(padding=2)(l11)
l13 = Conv1D(2,kernel_size = 4, strides = 1, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l12)
l14 = Conv1D(4,kernel_size = 2, strides = 2, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l13)
l15 = Conv1D(4,kernel_size = 1, strides = 1, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l14)
l16 = Conv1D(1,kernel_size = 1, strides = 1, activation = 'tanh', kernel_initializer = 'glorot_normal')(l15)
Eve_output = Flatten()(l16)

Alice_Bob_model = Model(inputs = [input_layer,auxillary_input],outputs = Bob_output)
Eve_model = Model(inputs = input_eve,outputs = Eve_output)

intermediate_layer_model = Model(inputs=Alice_Bob_model.input,outputs=Alice_Bob_model.get_layer('cipher_output').output)

#Let train_data is a matrix of shape mxno_input_neurons where for each training example has first bits for plaintext and next bits for key
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#Alice_Bob_model.compile(optimizer = adam, loss = loss_function(l8) , metrics =['accuracy'])
#Alice_Bob_model.compile(optimizer = adam, loss = loss_function(intermediate_layer_model.predict(Alice_Bob_model.input)) , metrics =['accuracy'])
Alice_Bob_model.compile(optimizer = adam, loss = custom_objective, metrics =['accuracy'])
#Eve_model.compile(optimizer = adam, loss = 'mean_absolute_error',metrics = ['accuracy'])
Eve_model.compile(optimizer = adam, loss = custom_objective2 , metrics = ['accuracy'])
Wsave = Eve_model.get_weights()

def average_reconstruction_loss(y_pred,y_true):
	error = np.absolute(y_pred - y_true)
	error = np.mean(error,axis=1)
	loss = np.zeros((int(y_pred.shape[0]/mini_batch_size)))
	for i in range(loss.shape[0]):
		loss[i] = np.mean(error[i*mini_batch_size:(i+1)*mini_batch_size])
	return loss
			
#AliceBob_error = []
#Eve_error = []
loss_AliceBob=[]
loss_Eve=[]
for k in range(NO_EPOCHS):
	#j=0
	no_iterations = 100
	print("RUN               : ", k+1)
	#while(j<train_data.shape[0]):
	r=0
	while(r<6667):
		j=0
		print("Batch ",r*3 +j+1," ------- ", r*3 + j + 3)
		train_data = get_data(m,no_input_neurons)
		# Training Alice-Bob
		print("Training Alice and Bob")
		data = train_data[j:j+mini_batch_size]
		y_output = train_data[j:j+mini_batch_size,0:plaintext_size]
		aux_data = train_data[j:j+mini_batch_size,plaintext_size:no_input_neurons]
		intermediate_output = intermediate_layer_model.predict([data,aux_data])
		# can pass the ciphertext i.e. intermediate_output by appending it to y_output and then split it back in the loss function
		history = LossHistory()
		Alice_Bob_model.fit([data,aux_data],y_output,batch_size = mini_batch_size,epochs = no_iterations,callbacks=[history])
		output = Alice_Bob_model.predict([data,aux_data])           # evaluate will calculate loss which also includes the Eve L1 error loss
		loss = average_reconstruction_loss(output,y_output)
		loss_AliceBob.append(loss[0])
		intermediate_output = intermediate_layer_model.predict([data,aux_data])
		output = Eve_model.predict(intermediate_output)           # evaluate will calculate loss which also includes the Eve L1 error loss
		loss = average_reconstruction_loss(output,y_output)
		loss_Eve.append(loss[0])		
		#minval = min(history.losses)
		#AliceBob_error.append(minval)
		j = j+mini_batch_size
			#Training Eve
		print("Training Eve")
		data = train_data[j:j+2*mini_batch_size]
		y_output = train_data[j:j+2*mini_batch_size,0:plaintext_size]
		aux_data = train_data[j:j+2*mini_batch_size,plaintext_size:no_input_neurons]
		intermediate_output = intermediate_layer_model.predict([data,aux_data])
		history = LossHistory()
		Eve_model.fit(intermediate_output,y_output,batch_size = mini_batch_size,epochs = no_iterations,callbacks = [history])
		output = Alice_Bob_model.predict([data,aux_data])           # evaluate will calculate loss which also includes the Eve L1 error loss
		loss = average_reconstruction_loss(output,y_output)
		loss_AliceBob.append(loss[0])
		loss_AliceBob.append(loss[1])
		intermediate_output = intermediate_layer_model.predict([data,aux_data])
		output = Eve_model.predict(intermediate_output)           # evaluate will calculate loss which also includes the Eve L1 error loss
		loss = average_reconstruction_loss(output,y_output)
		loss_Eve.append(loss[0])
		loss_Eve.append(loss[1])
		#minval = min(history.losses)
		#Eve_error.append(minval)
		j= j+2*mini_batch_size
		i = j
		r+=1
	i=0	
# plot the losses obtained
plt.plot(loss_AliceBob)
plt.plot(loss_Eve)
plt.legend(['bob', 'eve'])
plt.xlabel('Batch')
plt.ylabel('Lowest Decryption error achieved')
plt.show()

for k in range(5):
	Eve_model.set_weights(Wsave)
	for j in range(25,000):
		data = train_data
		aux_data = train_data[:,ciphertext_size:no_input_neurons]
		y_output = train_data[:,0:plaintext_size]
		intermediate_output = intermediate_layer_model.predict([data,aux_data])
		Eve_model.fit(intermediate_output,y_output,batch_size = mini_batch_size)
	#determine best Eve among the 5 and check if its accuracy increases

