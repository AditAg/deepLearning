import numpy as np
import cv2
import math
import random

output_size = 2
mean,standard_deviation = 0.0, 0.001                           # mean and s.d. for initialization of weights
learning_rate = 0.00001                                        # learning rate of SGD
no_sequences = 1000
# our input data is a matrix of n data points where each data point is basically a time sequence
def get_train_data(input_logs):                                # Here the input logs consist of 3 fields access time, unique cookie id and destination host name and it is sorted by the access time say it is a numpy array of shape no_input_logs x 3 
	dest_host_names = set()
	for i in range(input_logs.shape[0]):
		dest_host_names.add(input_logs[i][2])
	features = {}
	for i,p in enumerate(dest_host_names):
		features[p] = i
	
	train_data = np.zeros((input_logs.shape[0],len(features.keys())))
	for i in range(input_logs.shape[0]):
		val = features[input_logs[i][2]]
		train_data[val] = 1
	return train_data

class Recurrent_Neural_Network(object):

	def __init__(self,input_size,hidden_size,output_size,learning_rate,no_sequences):
		self.indim = input_size
		self.hdim = hidden_size
		self.outdim = output_size
		self.input_weights = np.random.normal(mean,standard_deviation,(self.indim,self.hdim))
		self.hidden_weights = np.random.normal(mean,standard_deviation,(self.hdim,self.hdim))
		self.output_weights = np.random.normal(mean,standard_deviation,(self.hdim,self.outdim))
		self.value_T = no_sequences
		self.hidden_input = np.zeros((self.value_T,self.hdim))
		self.hidden_output = np.zeros((self.value_T,self.hdim))
		self.output_inp = np.random.rand((self.value_T,self.outdim))
		self.output_out = np.random.rand((self.value_T,self.outdim))
		self.learning_rate = learning_rate
	
	def sigmoid(self,x):
		return (1.0/(1.0 + np.exp(-1.0 * x)))
	
	def softmax(self,x):
		e_x = np.exp(x - np.max(x))
		return (e_x/ e_x.sum())
	
	#def get_layers(self,input_data,hidden_output):       # input_data is a numpy array of size self.indim x 1 and hidden_output of self.hdim x 1
	#	val = np.dot(self.input_weights.T, input_data)
	#	val = val + np.dot(self.hidden_weights.T,hidden_output)
	#	hidden_layer = self.sigmoid(val)
	#	val2 = np.dot(self.output_weights.T,hidden_layer)
	#	output_layer = self.softmax(val2)
	#	return (val,hidden_layer,val2,output_layer)
	
	def get_layers(self,input_data):       # input_data is a numpy array of size T x self.indim
		for i in range(input_data.shape[0]):
			val = np.dot(self.input_weights.T, input_data[i])
			if(i==0):
				previous_hidden_out = np.zeros((self.hdim))
			else:
				previous_hidden_out = self.hidden_output[i-1]
			self.hidden_input[i] = val + np.dot(self.hidden_weights.T,previous_hidden_out)
			self.hidden_output[i] = self.sigmoid(self.hidden_input[i])
			self.output_inp[i] = np.dot(self.output_weights.T,self.hidden_output[i])
			self.output_out[i] = self.softmax(self.output_inp[i])
	
	def diff_sigmoid(self,x):
		z = self.sigmoid(x)
		return (z * (1.0-z))

	# Partial differential of Error function w.r.t input value of the unit in output layer i.e output_inp
	def diff_outinp(self,output,t):		    # output is a numpy array of shape (self.outdim,): each value in output is either 0 or 1 if it is 1 it means it should be the best unit e-g- if the 0th unit corresponds to label of fraudulent then for fraudulent input data the value of 0th unit should be 1.
		delE_vkt = self.output_out[t] - output
		return delE_vkt

	# Partial differential of Error function w.r.t input value of the unit in hidden layer i.e. hidden_input
	def diff_hiddeninput(self,ahead_diff_hiddeninput,t,output): # ahead_diff_hiddeninput represents diff_hidden_input for t+1
		delE_ujt = np.dot(self.output_weights,self.diff_outinp(output,t))
		delE_ujt = delE_ujt + np.dot(self.hidden_weights,ahead_diff_hiddeninput)
		delE_ujt = delE_ujt * self.diff_sigmoid(self.hidden_input[t])
		return delE_ujt
		
	def back_propagation(self,input_data,desired_output):  # desired_output is a numpy array of shape T x self.outdim and input_data T x self.indim
		#Partial differential of the error function w.r.t input_weights
		delE_input_weights = np.zeros((self.indim,self.hdim))
		for j in range(self.input_weights.shape[0]):
			ahead_diff_hiddeninput = np.zeros((self.hdim))
			for t in range((self.value_T)-1,-1,-1):
				val = self.diff_hiddeninput(ahead_diff_hiddeninput,t,desired_output[t])
				val = val*input_data[t][j]				
				if(t == (self.value_T)-1):
					sum = val
				else:
					sum = sum + val
				ahead_diff_hiddeninput = val
			delE_input_weights[j] = sum
		#Partial differential of the error function w.r.t hidden_weights
 		delE_hidden_weights = np.zeros((self.hdim,self.hdim))
		for j in range(self.hidden_weights.shape[0]):
			ahead_diff_hiddeninput = np.zeros((self.hdim))
			for t in range((self.value_T)-1,0,-1):
				val = self.diff_hiddeninput(ahead_diff_hiddeninput,t,desired_output[t])
				val = val*self.hidden_output[t-1][j]				
				if(t == (self.value_T)-1):
					sum = val
				else:
					sum = sum + val
				ahead_diff_hiddeninput = val
			delE_hidden_weights[j] = sum
		# Partial differential of the error function w.r.t output_weights
		delE_output_weights = np.zeros((self.hdim,self.outdim))
		for j in range(self.output_weights.shape[0]):
			for t in range(self.value_T):
				val = self.diff_outinp(desired_output[t],t)
				val = val*self.hidden_output[t][j]			
				if(t == 0):
					sum = val
				else:
					sum = sum + val
			delE_output_weights[j] = sum
		# Update the weights
		self.input_weights = self.input_weights - ((self.learning_rate)*(delE_input_weights))
		self.hidden_weights = self.hidden_weights - ((self.learning_rate)*(delE_hidden_weights))
		self.output_weights = self.output_weights - ((self.learning_rate)*(delE_output_weights))	
