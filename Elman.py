import cv2
import numpy as np
import random
import math
import sys

class Elman_Network(object):
	
	def __init__(self,no_epochs,learning_rate,no_input_neurons,hidden_size,output_size,context_size):
		self.indim = no_input_neurons
		self.no_epochs = no_epochs
		self.learning_rate = learning_rate
		self.hdim = hidden_size
		self.cdim = context_size
		self.outdim = output_size
		self.echo_step = 1
		self.batch_size = 100
		self.build_model()

	def tansig(self,x,deriv = False):
		val = ((2/(1.0 + np.exp(-2.0*x))) - 1)                   # can also do np.tanh(x)		
		if(deriv == True):
			return ((1+val) * (1+val) * np.exp(-2.0*x))
		return val
		
	def purelin(self,x,deriv = False):
		if(deriv==True):
			return np.ones_like(x)
		return x.copy()
		
	def _get_variable(self,name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
		if(weight_decay>0):
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		return tf.get_variable(name,shape = shape,initializer = initializer,dtype = dtype,regularizer = regularizer,trainable= trainable)
	
	def get_data(self):
		x = np.array(np.random.choice(2,self.seq_length))
		y = np.roll(x,self.echo_step)
		y[0:self.echo_step] = 0
		
		x = x.reshape((self.batch_size,-1))
		y = y.reshape((self.batch_size,-1))
		return (x,y)

	def build_model():
		self.weights_input_hidden = self._get_variable("weightsih",[self.indim+1,self.hdim],tf.contrib.layers.xavier_initializer())
		self.weights_context_hidden = self._get_variable("weightsch",[self.cdim+1,self.hdim],tf.contrib.layers.xavier_initializer())
		self.weights_hidden_output = self._get_variable("weightsho",[self.hdim+1,self.outdim],tf.contrib.layers.xavier_initializer())
		self.weights_hidden_context = self._get_variable("weightshc",[self.hdim+1,self.cdim],tf.contrib.layers.xavier_initializer())
		self.input_ = tf.placeholder("float",[None,self.indim])
		self.actual_output = tf.placeholder("float",[self.batch_size,self.indim])
		self.init_state = tf.placeholder(tf.float32,[self.batch_size,self.hdim])
		current_state = self.init_state
		inputs_series = tf.unstack(self.input_,axis=0)
		labels_series = tf.unstack(self.actual_output,axis=0)
		for current_input in inputs_series:
			current_input = tf.reshape(current_input,[None,1])
			
		tf.matmul(self.weights_input_hidden,tf.concat([self.input_,tf.constant(1,dtype=tf.float32,shape=[None,1])],1))
		tf.matmul(self.weights_context_hidden,tf.concat([self.context,tf.constant(1,dtype=tf.float32,shape=[None,1])],1))
		

