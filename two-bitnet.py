# To implement BinaryConnect --- Just change the discretization_function() to sign() and remove the optimal scaling factors
#To implement BWN --- Just change the discretization_function() to sign() and keep the scaling factors, now optimal scaling factor is obtained as:
#	alpha_l(t) = L1norm(W_{l}(t))/nl where nl is the number of weights in layer l
# To implement BinaryNet --- Change to sign() and add a sign() function to the output of each ReLu (or activation) function and also remove scaling factor
#To implement XNORNet ----- change to sign() and add a sign() function to outpout of activation function and update scaling factors as in BWN

from __future__ import division
import random
import cv2
import six
import tensorflow as tf
from keras import backend as K
import numpy as np


class Two_Bit_Network(object):

	def __init__(self,sess,input_shape = (224,224,1),outdim = 6,learning_rate = 0.01,epochs = 58,batch_size = 100,momentum=0.9,train_size = 1000):
		self.sess = sess
		self.input_shape = input_shape
		self.no_classes = outdim
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.momentum = momentum
		self.train_size = train_size
		self.build_model()
	
	def _get_variable(self,name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
		if(weight_decay>0):
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		return tf.get_variable(name,shape = shape,initializer = initializer,dtype = dtype,regularizer = regularizer,trainable= trainable)
	
	def discretize_function(self,Wi):
		if(Wi < -1):
			return -2
		elif(Wi <= 0):
			return -1
		elif(Wi <= 1):
			return 1
		else:
			return 2
	
	def conv2D(self,padding,input_,filter_shape,stride,name = "conv2d"):
		with tf.variable_scope(name):
			if('conv_layer' in name):
				w = self._get_variable('w',shape = filter_shape, initializer = tf.contrib.layers.xavier_initializer(),weight_decay = 0.0,dtype='float',trainable=False)
				w_binary = self._get_variable('binary_w',shape = filter_shape,initializer = tf.contrib.layers.xavier_initializer(),weight_decay = 0.0001,dtype='float',trainable = True)
				alpha = self._get_variable('scaling_factor',shape = filter_shape[3],initializer = tf.contrib.layers.xavier_initializer(),weight_decay =0.0, dtype = 'float',trainable = False)
				binarized_weights = alpha * w_binary
				conv = tf.nn.conv2d(input_,binarized_weights,stride,padding=padding)
			else:
				w = self._get_variable('w',shape = filter_shape,initializer = tf.contrib.layers.xavier_initializer(),weight_decay = 0.0001,dtype='float',trainable=True)
				conv = tf.nn.conv2d(input_,w,stride,padding = padding)
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

	def shortcut(self,input_,residual,i):
		input_shape = K.int_shape(input_)
		residual_shape = K.int_shape(residual)
		stride_width = int(round(input_shape[1]/residual_shape[1]))
		stride_height = int(round(input_shape[2]/residual_shape[2]))
		equal_channels = input_shape[3]==residual_shape[3]
		shortcut_val = input_
		if(stride_width > 1 or stride_height > 1 or not equal_channels):
			name1 = "shortcut_layer_" + str(i)
			shortcut_val = self.convolution(input_, filters = residual_shape[3],kernel_size = (1,1),name = name1,strides = (stride_width,stride_height),padding = "VALID")	
		return tf.add(shortcut_val,residual)	
	
	def basic_block(self,input_,filters,init_strides,i):
		name1 = "conv_layer_" + str(2*i+2)
		name2 = "conv_layer_" + str(2*i + 3 )
		if(i == 0):
			conv1 = self.convolution(input_,filters = filters,kernel_size = (3,3),name = name1,strides = init_strides,padding = "SAME")
		else:
			conv1 = self.batch_norm_relu(self.convolution(input_,filters= filters,kernel_size = (3,3),name = name1,strides = init_strides,padding = "SAME"))
		residual = self.batch_norm_relu(self.convolution(conv1,filters = filters,kernel_size=(3,3),name = name2,strides=(1,1),padding="SAME"))
		return self.shortcut(input_,residual,i)
		
	def build_model(self):
		self.input_ = tf.placeholder("float",[None,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
		self.conv1 = self.batch_norm_relu(self.convolution(self.input_,filters = 64,kernel_size = (7,7),name = "conv_layer_1",strides = (2,2),padding="SAME"))
		self.pool1 = self.pooling(self.conv1,pool_size = (3,3),strides = (2,2),type = "MAX")
		repetitions = 4
		filters = 64
		self.block = self.pool1
		for i in range(repetitions):
			init_strides = (1,1)
			if(i%2 == 0 and i!=0):
				init_strides = (2,2)
			self.block = self.basic_block(self.block,filters,init_strides,i)
			if(i%2 != 0):
				filters = filters *2
		self.block = self.batch_norm_relu(self.block)
		block_shape = K.int_shape(self.block)
		self.pool2 = self.pooling(self.block,pool_size=(block_shape[1], block_shape[2]),strides=(1, 1),type = "AVERAGE")
		self.flatten = tf.contrib.layers.flatten(self.pool2)
		flatten_shape = K.int_shape(self.flatten)		
		self.weights_FCLayer = self._get_variable("weights_final",[flatten_shape[1],self.no_classes],tf.contrib.layers.xavier_initializer())
		self.output = tf.nn.softmax(tf.matmul(self.flatten,self.weights_FCLayer))
		#print(K.int_shape(self.output))
		self.actual_output = tf.placeholder("float",[None,self.no_classes])		

	def generate_data(self):
		return (np.random.randint(0,256,size = (self.batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])), np.random.randint(2,size = (self.batch_size,self.no_classes)))
		
	def train(self):
		self.t_vars = tf.global_variables()
		self.original_weights = [var for var in self.t_vars if ('conv_layer' in var.name and '/w' in var.name)]
		self.binary_weights = [ var for var in self.t_vars if ('conv_layer' in var.name and 'binary' in var.name)]
		self.scaling_factors = [var for var in self.t_vars if 'scaling_factor' in var.name]		
		self.training_vars = tf.trainable_variables()
		self.loss_value = tf.reduce_mean(tf.abs(self.actual_output - self.output))
		for j in self.training_vars:
			print(j.name)
		#global_step = tf.Variable(0, trainable=False)
		#starter_learning_rate = 0.1
		#self.learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
		#self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum).compute_gradients(self.loss_value, var_list= self.training_vars,global_step = global_step)				
		self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum)
		tf.initialize_all_variables().run()
		for j in range(self.epochs):
			iterations = int(self.train_size/self.batch_size)
			print("Epoch",j+1)
			for k in range(iterations):
				print("Batch:",k+1)
				print("Determining binarized_weights and optimal scaling factor")
				if(j == 29 or j==39 or j==49):
					self.learning_rate = self.learning_rate/10
				
				for i in range(1,(len(self.original_weights)+1)):
					string = 'conv_layer_' + str(i)
					v = [v for v in self.original_weights if string in v.name][0]
					v2 = [v for v in self.binary_weights if string in v.name][0]
					scales = [v for v in self.scaling_factors if string in v.name][0]			
					v = v.eval(session = sess)
					sh = v.shape	
					v3 = np.zeros((sh))
					v4 = np.zeros((sh[3]))
					for l in range(sh[3]):
						sum1,count1 = 0.0,0
						sum2,count2 = 0.0,0
						for x in range(sh[0]):
							for y in range(sh[1]):
								for z in range(sh[2]):
									v3[x][y][z][l] = self.discretize_function(v[x][y][z][l])
									if(abs(v[x][y][z][l]) <=1):
										count1 = count1 +1
										sum1 = sum1 + abs(v[x][y][z][l])
									else:
										count2 = count2 + 1
										sum2 = sum2 + abs(v[x][y][z][l])
						val = ((sum1 + 2*sum2)*1.0/(count1 + 4*count2))
						v4[l] = val
					v2.assign(v3).eval()
					scales.assign(v4).eval()
				print("Training................")
				(images,output) = self.generate_data()
				vals = self.sess.run(self.optimizer.compute_gradients(self.loss_value, var_list= self.training_vars),feed_dict={self.actual_output:output,self.input_:images})      # Here vals is a list of (gradient,variable) pairs
				#for i in range(len(vals)):
				#	print ((vals[i][0],vals[i][1]))
				indices = []			
				for i in range(len(vals)):
					if(len(vals[i][1].shape)>=4 and (vals[i][1].shape[0] == vals[i][1].shape[1] != 1)):
						indices.append(i)
				#for i in range(len(indices)):
				#	weight = self.original_weights[i]
				#	vals[indices[i]] = list(vals[indices[i]])
				#	vals[indices[i]][1] = weight
				#	vals[indices[i]] = tuple(vals[indices[i]])
				#self.sess.run(self.optimizer.apply_gradients(vals),feed_dict={self.actual_output:output,self.input_:images})				
				for i in range(len(self.original_weights)):
					gradient = vals[indices[i]][0]
					weight = self.original_weights[i].eval()
					#print (gradient.shape,weight.shape)
					weight = weight - (self.learning_rate*(gradient))
					self.original_weights[i].assign(weight).eval()
				# apply gradient to the FC Layer weights and Shotcut Layer weights as well
				weight = self.weights_FCLayer.eval()
				#weight = ([v for v in self.t_vars if 'weights_final' in v.name][0]).eval()
				gradient = [x[0] for x in vals if (len(x[0].shape) == 2)][0]
				weight = weight - (self.learning_rate*gradient)
				self.weights_FCLayer.assign(weight).eval()
				weights = ([v for v in self.t_vars if 'shortcut_layer' in v.name])
				gradients = [x[0] for x in vals if (len(x[0].shape) == 4 and (x[0].shape[0] == 1))]
				for i in range(len(weights)):
					weight = weights[i].eval()
					weight = weight - (self.learning_rate*gradients[i])
					weights[i].assign(weight).eval()
	
sess = tf.Session()
with sess as sess:		
	network = Two_Bit_Network(sess)	
	network.train()

