# Idea is to have n different Neural Networks with same architecture for Alice-Bob,,,,, Eve has only 1 neural network for each of the n Alice-Bob networks.First take n different symmetric keys (here done by Diffie Hellman Key exchange)
# Then take for each Neural Network assign a particular symmetric Key
#Train each neural network on 20000 (say) batches and the single symmetric key it has been assigned with the Eve network remaining the same.
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

t = 10

def generate_polynomial(secret):
	vals = list(np.random.rand(1,t-1))
	vals = vals.append(secret)
	return np.poly1d(vals)

def polynomial_fit(x,y):                                        # x and y are points on the polynomial curve that we wish to estimate
	return np.poly1d(np.polyfit(x,y,t-1))                   # it returns the polynomial coefficients with least squared error.

def determine_secret(poly):
	return poly(0.0)

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


def custom_objective2(y_true,y_pred):
	return K.mean(tf.scalar_mul(1.0/m,(tf.reduce_sum(tf.abs(y_true - y_pred),1))))

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

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
Eve_model = Model(inputs = input_eve,outputs = Eve_output)
#Let train_data is a matrix of shape mxno_input_neurons where for each training example has first bits for plaintext and next bits for key
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#Alice_Bob_model.compile(optimizer = adam, loss = loss_function(l8) , metrics =['accuracy'])
#Alice_Bob_model.compile(optimizer = adam, loss = loss_function(intermediate_layer_model.predict(Alice_Bob_model.input)) , metrics =['accuracy'])
#Eve_model.compile(optimizer = adam, loss = 'mean_absolute_error',metrics = ['accuracy'])
Eve_model.compile(optimizer = adam, loss = custom_objective2 , metrics = ['accuracy'])


class Crypto_Network(object):
	def __init__(self,key):
		(self.Alice_Bob_model,self.intermediate_layer_model) = self.build_model()
		self.key = key
		self.Alice_Bob_model.compile(optimizer = adam, loss = self.custom_objective, metrics =['accuracy'])

	def custom_objective(self,y_true,y_pred):
		global i
		reconstruction_error = tf.scalar_mul(1.0/m,(tf.reduce_sum(tf.abs(y_true - y_pred),1)))
		data = train_data[i:i+mini_batch_size]
		aux_data = data[:,ciphertext_size:no_input_neurons]
		intermediate_output = self.intermediate_layer_model.predict([data,aux_data])
		eavesdropper_success = tf.scalar_mul(1.0/m,tf.reduce_sum(tf.abs(y_true-Eve_model.predict(intermediate_output)),1))
		tt = tf.constant(value = plaintext_size/2,shape = eavesdropper_success.get_shape())	
		eavesdropper_success = tf.square(tt - eavesdropper_success)
		eavesdropper_success = tf.scalar_mul((4.0/(plaintext_size**2)),(eavesdropper_success))
		return K.mean(reconstruction_error + eavesdropper_success)

	def build_model(self):
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
		Alice_Bob_model = Model(inputs = [input_layer,auxillary_input],outputs = Bob_output)
		intermediate_layer_model = Model(inputs=Alice_Bob_model.input,outputs=Alice_Bob_model.get_layer('cipher_output').output)
		return (Alice_Bob_model,intermediate_layer_model)
	
	def average_reconstruction_loss(self,y_pred,y_true):
		error = np.absolute(y_pred - y_true)
		error = np.mean(error,axis=1)
		loss = np.zeros((int(y_pred.shape[0]/mini_batch_size)))
		for i in range(loss.shape[0]):
			loss[i] = np.mean(error[i*mini_batch_size:(i+1)*mini_batch_size])
		return loss
			
	#AliceBob_error = []
	#Eve_error = []
	def train_model(self):
		loss_AliceBob=[]
		loss_Eve=[]
		for k in range(NO_EPOCHS):
			#j=0
			no_iterations = 100
			print("RUN               : ", k+1)
			#while(j<train_data.shape[0]):
			r=0
			while(r<500):
				j=0
				print("Batch ",r*3 +j+1," ------- ", r*3 + j + 3)
				train_data = get_data(m,plaintext_size)
				# Training Alice-Bob
				print("Training Alice and Bob")
				aux_data = np.tile(self.key,(mini_batch_size,1))
				data = np.concatenate((train_data[j:j+mini_batch_size],aux_data),axis=1)
				y_output = train_data[j:j+mini_batch_size]
				intermediate_output = self.intermediate_layer_model.predict([data,aux_data])
				self.Alice_Bob_model.fit([data,aux_data],y_output,batch_size = mini_batch_size,epochs = no_iterations)
				output = self.Alice_Bob_model.predict([data,aux_data]) 
				loss = self.average_reconstruction_loss(output,y_output)
				loss_AliceBob.append(loss[0])
				intermediate_output = self.intermediate_layer_model.predict([data,aux_data])
				output = Eve_model.predict(intermediate_output)
				loss = self.average_reconstruction_loss(output,y_output)
				loss_Eve.append(loss[0])		
				j = j+mini_batch_size
				
				print("Training Eve")
				aux_data = np.tile(self.key,(2*mini_batch_size,1))
				data = np.concatenate((train_data[j:j+2*mini_batch_size],aux_data),axis=1)
				y_output = train_data[j:j+2*mini_batch_size]
				intermediate_output = self.intermediate_layer_model.predict([data,aux_data])
				Eve_model.fit(intermediate_output,y_output,batch_size = mini_batch_size,epochs = no_iterations)
				output = self.Alice_Bob_model.predict([data,aux_data])
				loss = self.average_reconstruction_loss(output,y_output)
				loss_AliceBob.append(loss[0])
				loss_AliceBob.append(loss[1])
				intermediate_output = self.intermediate_layer_model.predict([data,aux_data])
				output = Eve_model.predict(intermediate_output)
				loss = self.average_reconstruction_loss(output,y_output)
				loss_Eve.append(loss[0])
				loss_Eve.append(loss[1])
				#minval = min(history.losses)
				#Eve_error.append(minval)
				j= j+2*mini_batch_size
				i = j
				r+=1
			i=0		
		#for k in range(5):
		#	Eve_model.set_weights(Wsave)
		#	for j in range(25,000):
		#		data = train_data
		#		aux_data = train_data[:,ciphertext_size:no_input_neurons]
		#		y_output = train_data[:,0:plaintext_size]
		#		intermediate_output = intermediate_layer_model.predict([data,aux_data])
		#		Eve_model.fit(intermediate_output,y_output,batch_size = mini_batch_size)
		#	#determine best Eve among the 5 and check if its accuracy increases
	
	def get_weights(self):
		return self.Alice_Bob_model.get_weights()

	def get_model(self):
		return self.Alice_Bob_model
#Ensemble and threshold cryptosystem
N = 100
models=[]
keys = np.random.randint(0,2,size = (N,key_size))*2 -1                       # N symmetric keys one for each neural network
for i in range(N):
	network = Crypto_Network(keys[i])
	network.train_model()
	mod = network.get_model()
	models.append(mod)

plaintext_message =np.random.randint(0,2,size = (1,plaintext_size))*2 -1
polynomials = []
for i in range(plaintext_message.shape[0]):
	polynomials.append(generate_polynomial(plaintext[i]))

# get n random points on each polynomial say they are in a list l of len 16 of n points at each element
xvals = np.random.random_sample([N])
l = []
for i in range(len(polynomials)):
	l1=[]
	for j in range(xvals.shape[0]):
		val = polynomials[i](xvals[j])
		l1.append((xvals[j],val)
	l.append(l1)
outputs = []
for i in range(N):
	input_vector=[]
	for j in range(len(l)):
		input_vector.append(l[j][i][0])
	input_vector = np.array(input_vector)
	output = models[i].predict(input_vector)
	for j in range(len(l)):
		output[j] = (output[j],l[j][i][1])
	outputs.append(output)
#outputs is a list of size N of 16 values
reconstructed_message= []
for i in range(len(l)):
	x = []
	y=[]
	for j in range(N):
		x.append(outputs[j][i][0])
		y.append(outputs[j][i][1])
	polynomial = polynomial_fit(x,y)
	reconstructed_message.append(determine_secret(polynomial))
