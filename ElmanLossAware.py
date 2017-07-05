import numpy as np
import cv2
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt

class Elman:
	
	def __init__(self,input_size,hidden_size,output_size,no_epochs,sess,learning_rate = 0.002,batch_size = 100,train_size =3000,beta1 = 0.9,beta2 = 0.9,epsilon = 1.0):
		self.indim = input_size
		self.hdim = hidden_size
		self.seq_length = self.indim
		self.outdim = output_size
		self.no_epochs = no_epochs
		self.echo_step = 1
		self.cdim = self.hdim
		self.train_size = train_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.sess = sess
		self.D = {}
		self.m1 = {}
		self.m2 = {} 
		self.initialize_matrices()
		self.build_model()

	def initialize_matrices(self):
		name = "weightsih"
		diag_matrix = np.random.uniform(-0.08,0.08,size=(self.indim+1,self.hdim))
		moments = np.random.random_sample(size = (self.indim+1,self.hdim))
		self.D[name] = diag_matrix
		self.m1[name] = moments
		self.m2[name] = moments
		name = "weightsch"
		diag_matrix = np.random.uniform(-0.08,0.08,size=(self.cdim+1,self.hdim))
		moments = np.random.random_sample(size = (self.cdim+1,self.hdim))
		self.D[name] = diag_matrix
		self.m1[name] = moments
		self.m2[name] = moments
		name = "weightsho"
		diag_matrix = np.random.uniform(-0.08,0.08,size=(self.hdim+1,self.outdim))
		moments = np.random.random_sample(size = (self.hdim+1,self.outdim))
		self.D[name] = diag_matrix
		self.m1[name] = moments
		self.m2[name] = moments
	
	def gen_data(self):
		x = np.array(np.random.choice(2,(self.train_size,self.seq_length)))
		y = np.zeros(x.shape)		
		for i in range(x.shape[0]-self.echo_step):
			for j in range(x.shape[1]):
				y[i][j] = y[i+self.echo_step][j]
		
		return (x,y)
		
	def _get_variable(self,name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
		if(weight_decay>0):
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		return tf.get_variable(name,shape = shape,initializer = initializer,dtype = dtype,regularizer = regularizer,trainable= trainable)
	
	def build_model(self):
		self.input_ = tf.placeholder(tf.float32,[self.batch_size,self.indim])
		self.actual_output = tf.placeholder(tf.float32,[self.batch_size,self.outdim])
		self.weights_input_hidden = self._get_variable("weightsih",[self.indim+1,self.hdim],tf.contrib.layers.xavier_initializer(), weight_decay=0.0,dtype='float',trainable=False)
		self.weights_context_hidden = self._get_variable("weightsch",[self.cdim+1,self.hdim],tf.contrib.layers.xavier_initializer(), weight_decay=0.0,dtype='float',trainable=False)
		self.weights_hidden_output = self._get_variable("weightsho",[self.hdim+1,self.outdim],tf.contrib.layers.xavier_initializer(), weight_decay=0.0,dtype='float',trainable=False)
		self.weights_input_hidden_binary = self._get_variable("bin_weightsih",[self.indim+1,self.hdim],tf.contrib.layers.xavier_initializer())
		self.weights_context_hidden_binary = self._get_variable("bin_weightsch",[self.cdim+1,self.hdim],tf.contrib.layers.xavier_initializer())
		self.weights_hidden_output_binary = self._get_variable("bin_weightsho",[self.hdim+1,self.outdim],tf.contrib.layers.xavier_initializer())
		self.alpha1 = self._get_variable("scaling_factor_ih",[1],tf.contrib.layers.xavier_initializer(), weight_decay=0.0,dtype='float',trainable=False)
		self.alpha2 = self._get_variable("scaling_factor_ch",[1],tf.contrib.layers.xavier_initializer(), weight_decay=0.0,dtype='float',trainable=False)
		self.alpha3 = self._get_variable("scaling_factor_ho",[1],tf.contrib.layers.xavier_initializer(), weight_decay=0.0,dtype='float',trainable=False)
		self.init_state = tf.placeholder(tf.float32,[1,self.hdim])
		self.context = self.init_state
		output_states=[]
		for i in range(self.batch_size):
			self.hidden = tf.sigmoid(tf.add(tf.matmul(tf.concat([tf.reshape(self.input_[i],[1,self.indim]),tf.constant(1,dtype=tf.float32,shape=[1,1])],1),(self.weights_input_hidden_binary * self.alpha1)),tf.matmul(tf.concat([self.context,tf.constant(1,dtype=tf.float32,shape=[1,1])],1),(self.weights_context_hidden_binary * self.alpha2))))
			self.output = tf.sigmoid(tf.matmul(tf.concat([self.hidden,tf.constant(1,dtype=tf.float32,shape=[1,1])],1), (self.weights_hidden_output_binary * self.alpha3)))
			output_states.append(self.output)
			self.context = tf.identity(self.hidden)
		losses = [tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels) for logits,labels in zip(output_states,tf.unstack(self.actual_output,axis=0))]
		self.total_loss = tf.reduce_mean(losses)

	def L1_norm(self,x):
		return np.sum(np.absolute(x))
	
	def element_wise_mult(self,x,y):
		return np.multiply(x,y)		
	
	def discretize_function(self,Wi):
		if(Wi>= 0 ):
			return 1
		else:
			return -1
	
	def train(self):
		self.t_vars = tf.global_variables()
		self.original_weights = [var for var in self.t_vars if ('weights' in var.name and '_' not in var.name)]
		self.binary_weights = [var for var in self.t_vars if('bin_weights' in var.name)]
		self.scaling_factors = [var for var in self.t_vars if('scaling_factor' in var.name)]
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta1,self.beta2,self.epsilon)
		self.training_vars = tf.trainable_variables()		
		tf.initialize_all_variables().run()
		(x,y) = self.gen_data()
		self.ov_loss = []
		for i in range(self.no_epochs):
			iterations = int(self.train_size/self.batch_size)
			self.initialize_matrices()
			print ("Epoch",i+1)
			self.losses = []
			for k in range(iterations):
				x1 = x[k*self.batch_size:(k+1)*self.batch_size]
				y1 = y[k*self.batch_size:(k+1)*self.batch_size]
				print("Batch:",k+1)
				print("Determining binarized_weights and optimal scaling factor")
				for j in range(len(self.original_weights)):
					v = self.original_weights[j]
					v2 = self.binary_weights[j]
					scales = self.scaling_factors[j]					
					v1 = v.eval(session = sess)
					sh = v1.shape	
					v3 = np.zeros((sh))
					v4 = np.zeros((1))
					#print(self.D[v.name[:-2]].shape)
					#print(v1.shape)
					for p in range(sh[0]):
						for q in range(sh[1]):
							v3[p][q] = self.discretize_function(v1[p][q])
					v4[0] = self.L1_norm(self.element_wise_mult(self.D[v.name[:-2]],v1))
					v4[0] = (v4[0]*1.0)/(self.L1_norm(self.D[v.name[:-2]])) 
					v2.assign(v3).eval()
					scales.assign(v4).eval()
				print("Training................")
				#self.loss_value = self.sess.run([self.total_loss],feed_dict={self.input_:x1, self.actual_output:y1, self.init_state:np.zeros((1,self.hdim))})
				loss,vals = self.sess.run([self.total_loss,self.optimizer.compute_gradients(self.total_loss, var_list= self.training_vars)],feed_dict={self.input_:x1, self.actual_output:y1, self.init_state:np.zeros((1,self.hdim))})
				self.losses.append(loss)
				vals = [(np.clip(grad,-5,5),var) for grad,var in vals]				
				for j in range(len(self.original_weights)):
					gradient = vals[j][0]
					weight = self.original_weights[j].eval()
					string = self.original_weights[j].name[:-2]
					self.m1[string] = self.beta1 * self.m1[string] + (1.0 - self.beta1)*gradient
					self.m2[string] = self.beta2 * self.m2[string] + (1.0 - self.beta2)*(self.element_wise_mult(gradient,gradient))
					m1_unbiased = ((self.m1[string]* 1.0)/(1.0 - self.beta1))
					m2_unbiased = ((self.m2[string]* 1.0)/(1.0 - self.beta2))
					self.D[string] = (1.0/self.learning_rate) * (self.epsilon + np.sqrt(m2_unbiased)) 
					weight = weight - (np.divide(m1_unbiased,self.D[string]))					
					weight = np.clip(weight,-1,1)
					self.original_weights[j].assign(weight).eval()			
			self.ov_loss.append((i+1,np.mean(self.losses)))
			if(i>=10):
				self.learning_rate = (self.learning_rate)/0.98
	
	def plot(self):
		x = []
		y = []
		for i in range(len(self.ov_loss)):
			x.append(self.ov_loss[i][0])
			y.append(self.ov_loss[i][1])
		plt.plot(x,y)
		plt.show()		
	
sess = tf.Session()
with sess as sess:
	network = Elman(256,512,256,200,sess)
	network.train()
	network.plot()
