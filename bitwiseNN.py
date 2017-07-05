import numpy as np
import cv2
import random
import tensorflow as tf

class BitWiseNeuralNetwork:
	
	def __init__(self,input_size,no_hidden_layers,hidden_layer_sizes,output_size,batch_size,learning_rate,sparsity_parameter,no_epochs):	
		self.indim = input_size
		self.no_hidden_layers = no_hidden_layers
		self.hidden_sizes = hidden_layer_sizes
		self.outdim = output_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.sparsity_parameter = sparsity_parameter
		self.tolerance_level = 0.001
		self.no_epochs = no_epochs
		self.initialize_weight_matrices()
	
	def initialize_weight_matrices(self):
		self.weights={}
		self.biases ={}
		for i in range(self.no_hidden_layers):
			name2 = "hidden" + str(i+1)
			name = "weights" + str(i+1)
			if(i==0):
				weights_matrix = np.random.randn(self.indim,self.hidden_sizes[0])
			else:
				weights_matrix = np.random.randn(self.hidden_sizes[i-1],self.hidden_sizes[i])
			#weights_matrix = np.clip(weights_matrix,-1,1)
			bias = np.random.randn(self.hidden_sizes[i])
			self.weights[name] = np.copy(weights_matrix)
			self.biases[name2] = np.copy(bias)
		i=i+1		
		name = "weights" + str(i+1)
		weights_matrix = np.random.randn(self.hidden_sizes[i-1],self.outdim)
		#weights_matrix = np.clip(weights_matrix,-1,1)
		self.weights[name] = np.copy(weights_matrix)
		name = "output"
		bias = np.random.randn(self.outdim)
		self.biases[name] = np.copy(bias)
		self.hidden_values = {}
		for i in range(self.no_hidden_layers):
			name = "hidden" + str(i+1)	
			self.hidden_values[name] = np.zeros((self.hidden_sizes[i]))

	def initial_activation(self,x):
		return np.tanh(x)
	
	def clipping_function(self,x):
		return np.tanh(x)

	def softmax(self,x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def forward_propagate(self,input_):
		self.input = input_
		for i in range(self.no_hidden_layers):
			name1 = "hidden" + str(i+1)
			name2 = "weights" + str(i+1)
			name3 = "hidden" + str(i)
			if(i==0):
				previous_layer = self.input
			else:
				previous_layer = self.hidden_values[name3]
			for j in range(self.hidden_values[name1].shape[0]):
				self.hidden_values[name1][j] = self.clipping_function(self.biases[name1][j])
				for k in range(self.weights[name2].shape[0]):
					self.hidden_values[name1][j]+=(self.clipping_function(self.weights[name2][k][j])*previous_layer[k])
				self.hidden_values[name1][j] = self.initial_activation(self.hidden_values[name1][j])
		previous_layer = self.hidden_values[str("hidden" + str(i+1))]
		weight = self.weights[str("weights" + str(i+2))]		
		self.output = np.zeros((self.outdim))
		for i in range(self.output.shape[0]):
			self.output[i] = self.clipping_function(self.biases["output"][i])
			for k in range(weight.shape[0]):
				self.output[i]+= (self.clipping_function(weight[k][i])*previous_layer[k])
		self.output = self.softmax(self.output)
	
	def loss_function(self,y,y1):
		return y
	
	def back_prop_weightcompress(self,input_,actual_output):
		self.gradients={}
		self.forward_propagate(input_)
		self.gradients["output"] = self.loss_function(self.output,actual_output)
		for i in range(self.no_hidden_layers,0,-1):
			name = "hidden" + str(i)
			weight = self.weights[str("weights" + str(i+1))]
			if(i==self.no_hidden_layers):
				ahead_gradient = self.gradients["output"]
			else:
				ahead_gradient = self.gradients[str("hidden"+str(i+1))]
			self.gradients[name] = np.zeros((self.hidden_sizes[i-1]))
			for j in range(self.hidden_sizes[i-1]):
				val = 0
				for k in range(weight.shape[1]):
					val = val + (self.clipping_function(weight[j][k])*ahead_gradient[k])
				self.gradients[name][j] = (val * (1 - np.square(self.hidden_values[name][j])))

		del_parameters = []
		for i in range(len(self.weights.keys())):
			name = "hidden" + str(i)
			name1 = "hidden" + str(i+1)
			if(i==0):
				previous_layer = self.input
			else:
				previous_layer = self.hidden_values[name]
			weight = self.weights[list(self.weights.keys())[i]]
			update_weight = np.zeros((weight.shape))
			update_bias = np.zeros((weight.shape[1]))
			for j in range(weight.shape[0]):
				for k in range(weight.shape[1]):
					val = 0
					if(i==len(self.weights.keys())-1):
						val = self.gradients["output"][k]*previous_layer[j]
						update_bias[k] = self.gradients["output"][k] * (1 - np.square(self.clipping_function(self.biases["output"][k])))
					else:
						val = self.gradients[name1][k]*previous_layer[j]
						update_bias[k] = self.gradients[name1][k]*(1- np.square(self.clipping_function(self.biases[name1][k])))
					val = val*(1-np.square(self.clipping_function(weight[j][k])))
					update_weight[j][k]+=val
			del_parameters.append(update_weight)
			del_parameters.append(update_bias)
		del self.gradients
		del ahead_gradient
		del update_weight
		del update_bias
		del weight
		return del_parameters
	
	def count_no_zeros(self,A,beta):
		return (np.absolute(A)<beta).sum()

	def binary_search(self,A,low,high,x):
		if(abs(high-low)<self.tolerance_level):
			return low
		if(low<high):
			mid = (low+high)/(2.0)
			z = self.count_no_zeros(A,mid)
			if(abs(z - x)<2):
				return mid
			elif(z < x):
				return self.binary_search(A,mid,high,x)
			else:
				return self.binary_search(A,low,mid,x)
		
	def initialize_params_bitwise(self):
		self.binarized_weights={}
		self.binarized_biases={}
		for i in self.weights.keys():
			weight = self.weights[i]
			minimum = np.amin(np.absolute(weight))
			maximum = np.amax(np.absolute(weight))
			count = int(self.sparsity_parameter * weight.shape[0] * weight.shape[1])
			beta = self.binary_search(weight,minimum,maximum,count)	
			for j in range(weight.shape[0]):
				for k in range(weight.shape[1]):
					if(weight[j][k]<(-1.0*beta)):
						weight[j][k] = -1
					elif(weight[j][k]>beta):
						weight[j][k] = 1
					else:
						weight[j][k] = 0
			self.binarized_weights[i] = weight
		for i in self.biases.keys():
			bias = self.biases[i]
			minimum = np.amin(np.absolute(bias))
			maximum = np.amax(np.absolute(bias))
			count = int(self.sparsity_parameter * bias.shape[0])
			beta = self.binary_search(bias,minimum,maximum,count)	
			for j in range(bias.shape[0]):
				if(bias[j]<(-1.0*beta)):
					bias[j] = -1
				elif(bias[j]>beta):
					bias[j] = 1
				else:
					bias[j] = 0
			self.binarized_biases[i] = bias
		del weight
		del minimum
		del maximum
		del bias

	def XNOR(self,x,y):
		if(x==0 or y==0):
			return 0
		if((x == -1 and y==-1 ) or (x == 1 and y == 1)):
			return 1
		else:
			return -1

	def activation_bitwise(self,x):
		if(x>=0):
			return 1
		else:
			return -1

	def feed_forward_bits(self,input_):
		self.input = input_
		for i in range(self.no_hidden_layers):
			name1 = "hidden" + str(i+1)
			name2 = "weights" + str(i+1)
			name3 = "hidden" + str(i)
			if(i==0):
				previous_layer = self.input
			else:
				previous_layer = self.hidden_values[name3]
			for j in range(self.hidden_values[name1].shape[0]):
				self.hidden_values[name1][j] = self.binarized_biases[name1][j]
				for k in range(self.binarized_weights[name2].shape[0]):
					self.hidden_values[name1][j]+=self.XNOR(self.binarized_weights[name2][k][j],previous_layer[k])
				self.hidden_values[name1][j] = self.activation_bitwise(self.hidden_values[name1][j])
		previous_layer = self.hidden_values[str("hidden" + str(i+1))]
		weight = self.binarized_weights[str("weights" + str(i+2))]		
		self.output = np.zeros((self.outdim))
		for i in range(self.output.shape[0]):
			self.output[i] = self.binarized_biases["output"][i]
			for k in range(weight.shape[0]):
				self.output[i]+= self.XNOR(weight[k][i],previous_layer[k])
			self.output[i] = self.activation_bitwise(self.output[i])		

	def loss_function_bitwise(self,pred,act):
		error = np.zeros((act.shape))
		for i in range(act.shape[0]):
			error[i]= ((1 - self.XNOR(act[i],pred[i]))/2)
		return error		
	
	def backprop_bitwise(self,input_,actual_output):
		#print(input_,actual_output)
		self.gradients={}
		self.feed_forward_bits(input_)
		self.loss_value = self.loss_function_bitwise(self.output,actual_output)
		self.gradients["output"] = -(actual_output/(2.0))   #### This is the derivative of the used prediction errori.e. loss function w.r.t. the final layer output,(error = (1- targetXNORoutput)/2)
		for i in range(self.no_hidden_layers,0,-1):
			name = "hidden" + str(i)
			weight = self.binarized_weights[str("weights" + str(i+1))]
			if(i==self.no_hidden_layers):
				ahead_gradient = self.gradients["output"]
			else:
				ahead_gradient = self.gradients[str("hidden"+str(i+1))]
			self.gradients[name] = np.zeros((self.hidden_sizes[i-1]))
			#print(weight,ahead_gradient)
			for j in range(self.hidden_sizes[i-1]):
				val = 0
				for k in range(weight.shape[1]):
					val = val + (weight[j][k]*ahead_gradient[k])
				self.gradients[name][j] = val

		del_parameters = []
		for i in range(len(self.binarized_weights.keys())):
			name = "hidden" + str(i)
			name1 = "hidden" + str(i+1)
			if(i==0):
				previous_layer = self.input
			else:
				previous_layer = self.hidden_values[name]
			weight = self.binarized_weights[list(self.binarized_weights.keys())[i]]
			update_weight = np.zeros((weight.shape))
			update_bias = np.zeros((weight.shape[1]))
			for j in range(weight.shape[0]):
				for k in range(weight.shape[1]):
					val = 0
					if(i==len(self.binarized_weights.keys())-1):
						val = self.gradients["output"][k]*previous_layer[j]
						update_bias[k] = self.gradients["output"][k] 
					else:
						val = self.gradients[name1][k]*previous_layer[j]
						update_bias[k] = self.gradients[name1][k]
					update_weight[j][k]+=val
			del_parameters.append(update_weight)
			del_parameters.append(update_bias)
		del self.gradients
		del ahead_gradient
		del update_weight
		del update_bias
		del weight
		return del_parameters	
	
	def train(self,input_,output_,bit):
		no_batches = int(input_.shape[0]/(self.batch_size))
		for p in range(self.no_epochs):
			print("Epoch",p+1)
			for i in range(no_batches):
				print("Batch",i+1)
				start = i*self.batch_size
				self.initialize_params_bitwise()
				for k in range(self.batch_size):
					data = input_[start+k]
					outp = output_[start+k]
					if(bit==0):
						del_parameters = self.back_prop_weightcompress(data,outp)
					else:
						del_parameters = self.backprop_bitwise(data,outp)
					if(k==0):
						gradient_parameters = del_parameters[:]
					else:
						gradient_parameters = [np.add(x1,x2) for x1,x2 in zip(gradient_parameters,del_parameters)]
				
				for k in range(len(gradient_parameters)):
					if(k%2==0):
						name = "weights" + str(int(k/2)+1)
						self.weights[name] = self.weights[name] -self.learning_rate*gradient_parameters[k]
					else:
						name = "hidden" + str(int(k/2)+1)
						if(k == len(gradient_parameters) -1):
							name = "output"
						self.biases[name] = self.biases[name] - self.learning_rate*gradient_parameters[k]
		del gradient_parameters		

network = BitWiseNeuralNetwork(128,3,(256,256,256),6,10,0.01,0.5,10)
arr=[-1,1]
network.train(np.random.choice(arr,(100,128)),np.random.choice(arr,(100,6)),0)
network.train(np.random.choice(arr,(100,128)),np.random.choice(arr,(100,6)),1)
