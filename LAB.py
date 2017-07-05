#Make LAB for recurrent-LSTM, CIFAR-10, MNIST
import random
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

class LAB(object):
	
	def __init__(self,sess,no_layers = 3,input_size = (224,224,1), output_size = 6,learning_rate = 0.01,batch_size = 10,train_size =100,no_epochs = 10,beta1 = 0.9,beta2 = 0.9,epsilon = 1.0):
		self.no_layers = no_layers
		self.input_shape = input_size
		self.no_classes = output_size
		self.learning_rate = learning_rate
		self.filters = []
		val = 64
		self.batch_size = batch_size
		self.no_epochs = no_epochs
		self.train_size = train_size
		self.filter_size = [(7,7),(3,3),(3,3)]
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.sess = sess
		for i in range(self.no_layers):
			self.filters.append(val)
			val = int(val/2)
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
	
	def conv2D(self,padding,input_,filter_shape,stride,name = "conv2d"):
		with tf.variable_scope(name):
			w = self._get_variable('w',shape = filter_shape, initializer = tf.contrib.layers.xavier_initializer(),weight_decay = 0.0,dtype='float',trainable=False)
			w_binary = self._get_variable('binary_w',shape = filter_shape,initializer = tf.contrib.layers.xavier_initializer(),weight_decay = 0.0001,dtype='float',trainable = True)
			alpha = self._get_variable('scaling_factor',shape = filter_shape[3],initializer = tf.contrib.layers.xavier_initializer(),weight_decay =0.0, dtype = 'float',trainable = False)
			binarized_weights = alpha * w_binary
			conv = tf.nn.conv2d(input_,binarized_weights,stride,padding=padding)
			return conv
	
	def batch_norm_relu(self,input_):
		p = tf.nn.moments(input_,axes = [0],keep_dims = False)
		norm = tf.nn.batch_normalization(input_,p[0],p[1],offset=None,scale=None,variance_epsilon =1e-4)
		return tf.nn.relu(norm)
	
	def convolution(self,input_,filters,kernel_size,name,strides=(1,1),padding = "SAME"):
		kernel_size = kernel_size + (input_.shape[3],filters)
		strides = [1,strides[0],strides[1],1]
		return self.conv2D(padding,input_, filter_shape=kernel_size,stride=strides,name=name)		
	
	def pooling(self,input_,pool_size,strides,type="MAX"):
		pool_size = [1,pool_size[0],pool_size[1],1]
		strides = [1,strides[0],strides[0],1]
		if(type == "MAX"):
			return tf.nn.max_pool(input_,pool_size,strides,padding = "SAME")
		elif(type == "AVERAGE"):
			return tf.nn.avg_pool(input_,pool_size,strides,padding = "SAME")

	def build_model(self):
		self.input_ = tf.placeholder("float",[None,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
		for i in range(self.no_layers):
			if(i==0):
				self.conv = self.batch_norm_relu(self.convolution(self.input_,filters = self.filters[i],kernel_size = self.filter_size[i],name = "conv_layer_"+str(i+1),strides = (2,2),padding="SAME"))
			else:
				self.conv = self.batch_norm_relu(self.convolution(self.pool,filters = self.filters[i],kernel_size = self.filter_size[i],name = "conv_layer_"+str(i+1),strides = (1,1),padding = "SAME"))
			self.pool = self.pooling(self.conv,pool_size = (2,2),strides = (2,2),type = "MAX")
		self.flatten = tf.contrib.layers.flatten(self.pool)
		flatten_shape = K.int_shape(self.flatten)		
		self.weights_FCLayer = self._get_variable("weights_final",[flatten_shape[1],self.no_classes],tf.contrib.layers.xavier_initializer())
		self.output = tf.nn.softmax(tf.matmul(self.flatten,self.weights_FCLayer))
		#print(K.int_shape(self.output))
		self.actual_output = tf.placeholder("float",[None,self.no_classes])	

	def generate_data(self):
		return (np.random.randint(0,256,size = (self.batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])), np.random.randint(2,size = (self.batch_size,self.no_classes)))
		
	def L1_norm(self,x):
		return np.sum(np.absolute(x))
	
	def element_wise_mult(self,x,y):
		return np.multiply(x,y)		
	
	def update_learning_rate(self,x):
		return x
	def train(self):
		self.t_vars = tf.global_variables()
		self.original_weights = [var for var in self.t_vars if ('conv_layer' in var.name and '/w' in var.name)]
		self.binary_weights = [ var for var in self.t_vars if ('conv_layer' in var.name and 'binary' in var.name)]
		self.scaling_factors = [var for var in self.t_vars if 'scaling_factor' in var.name]		
		self.training_vars = tf.trainable_variables()
		self.loss_value = tf.reduce_mean(tf.abs(self.actual_output - self.output))
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta1,self.beta2,self.epsilon)
		tf.initialize_all_variables().run()
		for i in range(self.no_epochs):
			iterations = int(self.train_size/self.batch_size)
			self.initialize_matrices()
			print("Epoch",i+1)
			for k in range(iterations):
				print("Batch:",k+1)
				print("Determining binarized_weights and optimal scaling factor")
				for j in range(1,(len(self.original_weights)+1)):
					string = 'conv_layer_' + str(j)
					v = [v for v in self.original_weights if string in v.name][0]
					v2 = [v for v in self.binary_weights if string in v.name][0]
					scales = [v for v in self.scaling_factors if string in v.name][0]			
					v = v.eval(session = sess)
					sh = v.shape	
					v3 = np.zeros((sh))
					v4 = np.zeros((sh[3]))
					for l in range(sh[3]):
						for x in range(sh[0]):
							for y in range(sh[1]):
								for z in range(sh[2]):
									v3[x][y][z][l] = self.discretize_function(v[x][y][z][l])
						v4[l] = self.L1_norm(self.element_wise_mult(self.D[string][:,:,:,l],v[:,:,:,l]))
						v4[l] = (v4[l]*1.0)/(self.L1_norm(self.D[string][:,:,:,l])) 
					v2.assign(v3).eval()
					scales.assign(v4).eval()
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
					#Weight regularization ---- Clip the real-valued weights as well to prevent them from growing very large
					self.original_weights[j].assign(weight).eval()
				weight = self.weights_FCLayer.eval()
				gradient = [x[0] for x in vals if (len(x[0].shape) == 2)][0]
				weight = weight - (self.learning_rate*gradient)
				self.weights_FCLayer.assign(weight).eval()
				self.learning_rate = self.update_learning_rate(self.learning_rate)
		
sess = tf.Session()
with sess as sess:
	network = LAB(sess)
	network.train()
		
