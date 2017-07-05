import numpy as np
import cv2
import tensorflow as tf

class Network:
	
	def __init__(self,input_size,no_hidden_layers,hidden_sizes,learning_rate,beta1,beta2,epsilon,no_epochs,batch_size,train_size,sess):
		self.indim = input_size
		self.no_classes = 2
		self.no_layers=no_hidden_layers
		self.hidden_sizes = hidden_sizes
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.no_epochs = no_epochs
		self.batch_size = batch_size
		self.train_size = train_size
		self.sess = sess
		self.D = {}
		self.m1 = {}
		self.m2 = {} 
		self.initialize_matrices()
		self.build_model()

	def initialize_matrices(self):
		for i in range(self.no_layers+1):
			name = "hidden"+str(i+1)
			if(i==self.no_layers):
				name = "output"
				size = (self.hidden_sizes[i-1],self.no_classes)
			elif(i==0):
				size = (self.indim,self.hidden_sizes[i])
			else:
				size = (self.hidden_sizes[i-1],self.hidden_sizes[i])
			diag_matrix = np.random.normal(0,0.1,size)
			moments = np.random.random_sample(size)
			self.D[name] = diag_matrix
			self.m1[name] = moments
			self.m2[name] = moments
			
	def _get_variable(self,name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
		if(weight_decay>0):
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		return tf.get_variable(name,shape = shape,initializer = initializer,dtype = dtype,regularizer = regularizer,trainable= trainable)
	
	def forward_layer(self,input_,hidden_size,name="hidden"):
		shape = (input_.shape[1],hidden_size)
		with tf.variable_scope(name):
			w = self._get_variable('w',shape,tf.contrib.layers.xavier_initializer(),weight_decay=0.0,dtype='float',trainable=False)
			binary_w = self._get_variable('binary_w',shape,tf.contrib.layers.xavier_initializer(),weight_decay=0.0, dtype='float', trainable=True)
			scaling_factor = self._get_variable('scaling_factor',[1],tf.contrib.layers.xavier_initializer(), weight_decay=0.0, dtype='float', trainable=False)
			bias = self._get_variable('bias',[hidden_size],tf.contrib.layers.xavier_initializer())
			binarized_weights = scaling_factor * binary_w
			hidden_output = tf.sigmoid(tf.add(tf.matmul(input_,binarized_weights),bias))
			return hidden_output
		
	def build_model(self):
		self.input_ = tf.placeholder(tf.float32,[None,self.indim])
		for i in range(self.no_layers+1):
			if(i==0):
				self.hidden_output = self.forward_layer(self.input_,self.hidden_sizes[i],"hidden"+str(i+1))
			elif(i==self.no_layers):
				self.output = self.forward_layer(self.hidden_output,self.no_classes,"output")
			else:
				self.hidden_output = self.forward_layer(self.hidden_output,self.hidden_sizes[i],"hidden" + str(i+1))
		self.output = tf.nn.softmax(self.output)
		self.actual_output = tf.placeholder(tf.float32,[None,self.no_classes])
	
	def discretize_function(self,Wi):
		if(Wi>= 0 ):
			return 1
		else:
			return -1
	
	def L1_norm(self,x):
		return np.sum(np.absolute(x))
	
	def element_wise_mult(self,x,y):
		return np.multiply(x,y)		
	
	def update_learning_rate(self,x,i):
		if(i%15==0 and i!=0):
			return ((x*1.0)/0.5)
		return x
		
	def train(self,input_,output_):
		self.t_vars = tf.trainable_variables()
		self.all_vars = tf.global_variables()
		self.original_weights = [var for var in self.all_vars if '/w' in var.name]
		self.binary_weights = [var for var in self.all_vars if('/binary_w' in var.name)]
		self.scaling_factors = [var for var in self.all_vars if('/scaling_factor' in var.name)]
		self.biases = [var for var in self.all_vars if('/bias' in var.name)]
		self.loss_value = tf.reduce_mean(tf.abs(self.actual_output - self.output))
		
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta1,self.beta2,self.epsilon)
		tf.initialize_all_variables().run()
		for j in range(self.no_epochs):
			iterations = int(self.train_size/self.batch_size)
			print("Epoch",j+1)
			for k in range(iterations):
				print("Batch:",k+1)
				print("Determining binarized_weights and optimal scaling factor")
				data = input_[k*self.batch_size:(k+1)*self.batch_size]
				outp = output_[k*self.batch_size:(k+1)*self.batch_size]
				for i in range(1,(len(self.original_weights)+1)):
					string = 'hidden' + str(i)
					if(i==len(self.original_weights)):
						string = 'output'
					v1 = [v for v in self.original_weights if string in v.name][0]
					v2 = [v for v in self.binary_weights if string in v.name][0]
					scales = [v for v in self.scaling_factors if string in v.name][0]			
					v1 = v1.eval(session = sess)
					sh = v1.shape
					v3 = np.zeros((sh))
					v4 = np.zeros((1))
					for x in range(sh[0]):
						for y in range(sh[1]):
							v3[x][y] = self.discretize_function(v1[x][y])
					v4[0] = self.L1_norm(self.element_wise_mult(self.D[string],v1))
					v4[0] = (v4[0]*1.0)/(self.L1_norm(self.D[string])) 
					v2.assign(v3).eval()
					scales.assign(v4).eval()
				print("Training................")
				vals = self.sess.run(self.optimizer.compute_gradients(self.loss_value, var_list= self.t_vars),feed_dict={self.actual_output:outp,self.input_:data})  
				indices=[] 
				for i in range(len(vals)):
					if(len(vals[i][1].shape)==2):
						indices.append(i)
				for i in range(len(self.original_weights)):
					gradient = vals[indices[i]][0]
					weight = self.original_weights[i].eval()
					#print (gradient.shape,weight.shape)
					if(i==len(self.original_weights)-1):
						string = 'output'
					else:
						string = 'hidden'+str(i+1)
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
					self.original_weights[i].assign(weight).eval()
				for i in range(len(vals)):
					if(i%2==1):
						t = int(i/2)
						bias = self.biases[t].eval()
						bias = bias - vals[i][0]
						self.biases[t].assign(bias).eval()
			self.learning_rate = self.update_learning_rate(self.learning_rate,j)
	
	def test(self,input_):
		output = self.sess.run(self.output,feed_dict={self.input_:input_})
		print(output)				

sess = tf.Session()
with sess as sess:
	network = Network(4,3,(10,10,10),0.01,0.9,0.9,0.9,10,10,100,sess)	
	network.train(np.random.choice(2,(100,4)),np.random.choice(2,(100,2)))	
	network.test(np.random.choice(2,(1,4)))	
