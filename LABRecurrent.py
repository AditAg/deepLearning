#Make LAB for recurrent-LSTM, CIFAR-10, MNIST
import random
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

class LAB(object):
	
	def __init__(self,sess,no_layers = 6,input_size = (32,32,3), output_size = 10,learning_rate = 0.01,batch_size = 10,train_size =100,no_epochs = 10,beta1 = 0.9,beta2 = 0.9,epsilon = 1.0):
		self.no_layers = no_layers
		self.input_shape = input_size
		self.no_classes = output_size
		self.lstm_size = 512
		self.learning_rate = learning_rate
		self.filters = []
		val = 128
		self.batch_size = batch_size
		self.no_epochs = no_epochs
		self.train_size = train_size
		self.filter_size = [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)]
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.sess = sess
		for i in range(self.no_layers):
			self.filters.append(val)
			self.filters.append(val)
			val = int(val*2)
		self.D = {}
		self.m1 = {}
		self.m2 = {} 
		self.initialize_matrices()
		self.build_model()
	
	def initialize_matrices(self):
		for i in range(self.no_layers):
			if(i==0):
				size = self.filter_size[i] + (self.input_shape[2],self.filters[i])
			else:
				size = self.filter_size[i] + (self.filters[i-1],self.filters[i])
			diag_matrix = np.random.normal(0,0.1,size)
			moments = np.random.random_sample(size)
			name = 'conv_layer_' + str(i+1)
			self.D[name] = diag_matrix
			self.m1[name] = moments
			self.m2[name] = moments
	
	def _get_variable(self,name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
		if(weight_decay>0):
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		return tf.get_variable(name,shape = shape,initializer = initializer,dtype = dtype,regularizer = regularizer,trainable= trainable)
	
	def discretize_function(self,Wi):
		if(Wi>= 0 ):
			return 1
		else:
			return -1
	
	def batch_norm_relu(self,input_):
		p = tf.nn.moments(input_,axes = [0],keep_dims = False)
		norm = tf.nn.batch_normalization(input_,p[0],p[1],offset=None,scale=None,variance_epsilon =1e-4)
		return tf.nn.relu(norm)
	
	def build_model(self):
		self.input_ = tf.placeholder("float",[None,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
		self.lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
		# Initial state of the LSTM memory.
		self.val,self.state = tf.nn.dynamic_rnn(self.lstm,self.input_)
		self.val = tf.transpose(self.val, [1, 0, 2])
		self.last = tf.gather(self.val, int(val.get_shape()[0]) - 1)
		self.weights_layer = self._get_variable("weights_layer",[self.lstm_size,self.no_classes],tf.contrib.layers.xavier_initializer())
		self.output = tf.nn.softmax(tf.matmul(self.last,self.weights_layer))
		self.actual_output = tf.placeholder("float",[None,self.no_classes])	

	def generate_data(self):
		return (np.random.randint(0,256,size = (self.batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])), np.random.randint(2,size = (self.batch_size,self.no_classes)))
		
	def L1_norm(self,x):
		return np.sum(np.absolute(x))
	
	def element_wise_mult(self,x,y):
		return np.multiply(x,y)		
	
	def update_learning_rate(self,x,i):
		if(i%15==0 and i!=0):
			return ((x*1.0)/0.5)
		return x
		
	def train(self):
		self.t_vars = tf.global_variables()
		self.loss_value = -tf.reduce_sum(self.actual_output *(tf.log(tf.clip_by_value(self.output,1e-10,1.0))))
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta1,self.beta2,self.epsilon)
		tf.initialize_all_variables().run()
		for i in range(self.no_epochs):
			iterations = int(self.train_size/self.batch_size)
			self.initialize_matrices()
			print("Epoch",i+1)
			for k in range(iterations):
				print("Batch:",k+1)
				
				print("Training................")
				(images,output) = self.generate_data()
				vals = self.sess.run(self.optimizer.compute_gradients(self.loss_value, var_list= self.training_vars),feed_dict={self.actual_output:output,self.input_:images})      # Here vals is a list of (gradient,variable) pairs
				indices = []			
				for j in range(len(vals)):
					if(len(vals[j][1].shape)>=4):
						indices.append(j)
				for j in range(len(self.original_weights)):
					gradient = vals[indices[j]][0]
					weight = self.original_weights[j].eval()
					#print (gradient.shape,weight.shape)
					string = 'conv_layer_' + str(j+1)
					self.m1[string] = self.beta1 * self.m1[string] + (1.0 - self.beta1)*gradient
					self.m2[string] = self.beta2 * self.m2[string] + (1.0 - self.beta2)*(self.element_wise_mult(gradient,gradient))
					m1_unbiased = ((self.m1[string]* 1.0)/(1.0 - self.beta1))
					m2_unbiased = ((self.m2[string]* 1.0)/(1.0 - self.beta2))
					#print(m2_unbiased)
					#abc = input()
					self.D[string] = (1.0/self.learning_rate) * (self.epsilon + np.sqrt(m2_unbiased)) 
					#self.D[string] = np.nan_to_num(self.D[string])
					#print(self.D[string])
					#abc = input()
					weight = weight - (np.divide(m1_unbiased,self.D[string]))
					self.original_weights[j].assign(weight).eval()
				weight = self.weights_FCLayer1.eval()
				gradient = [x[0] for x in vals if (len(x[0].shape) == 2)][0]
				weight = weight - (self.learning_rate*gradient)
				self.weights_FCLayer1.assign(weight).eval()
				weight = self.weights_FCLayer2.eval()
				gradient = [x[0] for x in vals if (len(x[0].shape) == 2)][1]
				weight = weight - (self.learning_rate*gradient)
				self.weights_FCLayer2.assign(weight).eval()
				weight = self.weights_FCLayer.eval()
				gradient = [x[0] for x in vals if (len(x[0].shape) == 2)][2]
				weight = weight - (self.learning_rate*gradient)
				self.weights_FCLayer.assign(weight).eval()
				self.learning_rate = self.update_learning_rate(self.learning_rate,i)
		
sess = tf.Session()
with sess as sess:
	network = LAB(sess)
	network.train()
		
