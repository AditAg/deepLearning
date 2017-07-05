#considering even parity
import cv2
import random
import numpy as np

no_output_bits = 100000
N = 100
k = 20
ita_STDP = 0.01 
ita_IP = 0.01
change_parity_block = 4                           # After after every 4 timesteps the parity block is moved q steps along the state vector
q = 5

class Binary_Rec_Network(object):
	
	def __init__(self,N,k,decrease_rate_antiSTDP,learning_rateIP,q):
		self.no_neurons = N
		self.no_active_neurons = k
		self.Weights = np.random.normal(0,0.1,(self.no_neurons,self.no_neurons))
		for i in range(self.Weights.shape[0]):
			self.Weights[i][i] = 0.0
		self.thresholds = np.random.normal(0,0.1,(self.no_neurons))
		self.decrease_rate_antiSTDP = decrease_rate_antiSTDP
		self.learning_rateIP = learning_rateIP
		self.start = 0
		self.q = q
		self.Weights_p = np.random.normal(0,0.1,(self.no_neurons+self.q+1,self.no_neurons))
		self.threshold_p = np.random.normal(0,0.1,(self.no_neurons+self.q+1))
		self.state_vector_p = np.zeros((self.no_neurons))
	
	def set_state_vector(self,vector):
		self.state_vector_p = np.copy(vector)

	def get_weightsp(self):
		Weights_p = np.zeros((self.no_neurons+1+self.q,self.no_neurons))
		for i in range(self.Weights.shape[0]):
			for j in range(self.Weights.shape[1]):
				Weights_p[i][j] = self.Weights[i][j]
		for i in range(self.q):
			for j in range(self.q):
				Weights_p[i+self.no_neurons][j] = 1
		val = 1		
		for i in range(self.q):
			Weights_p[self.no_neurons+self.q][i] = val
			val = val *(-1)
		self.Weights_p = Weights_p

	def get_thresholdsp(self):
		thresholds_p = np.zeros((self.no_neurons + self.q + 1))
		for i in range(self.no_neurons):
			thresholds_p[i] = self.thresholds[i]
		for i in range(self.q):
			thresholds_p[i+self.no_neurons] = i+1
		thresholds_p[self.no_neurons+self.q] = 0
		self.threshold_p = thresholds_p

	def random_initial_state(self):
		x0 = np.random.random_sample((self.no_neurons))
		idx = np.argpartition(x0,-self.no_active_neurons)
		p = idx.shape[0]
		for i in range(p):
			if(i>=(p - self.no_active_neurons)):
				x0[idx[i]] = 1
			else:
				x0[idx[i]] = 0
		del idx
		del p 
		return x0
		
	def update_function(self,previous_state,current_state):
		pre_activations = np.zeros((self.no_neurons))
		for i in range(pre_activations.shape[0]):
			sum = 0
			for j in range(self.Weights.shape[1]):
				sum += (self.Weights[i][j] * current_state[j])
			sum = sum - self.thresholds[i]
			sum = sum - max(current_state[i],previous_state[i])
			pre_activations[i] = sum
		new_state = np.zeros((self.no_neurons))
		idx = np.argpartition(pre_activations,-self.no_active_neurons)
		p = idx.shape[0]
		for i in range(p):
			if(i>=(p - self.no_active_neurons)):
				new_state[idx[i]] = 1
			else:
				new_state[idx[i]] = 0
		del p
		del idx
		del pre_activations
		return new_state
	
	def update_weights_antiSTDP(self,current_state,previous_state):
		for i in range(self.Weights.shape[0]):
			for j in range(self.Weights.shape[1]):
				delWij = -1.0 * self.decrease_rate_antiSTDP*((current_state[i]*previous_state[j]) - (previous_state[i]*current_state[j]))	
				self.Weights[i][j] = self.Weights[i][j] + delWij

	def update_thresholds_IP(self,current_state):
		# The average activity of a given unit in the network is driven towards k/N i.e. each neuron will fire on an average k out of every N timesteps		
		self.thresholds = self.thresholds + self.learning_rateIP * (current_state - (self.no_active_neurons/self.no_neurons)) 
	
	def determine_parity(self,t,current_state,previous_state):
		if(t%change_parity_block == 0):
			self.start = self.start+self.q	
		if(self.start > self.no_neurons - self.q):
			self.start = 0	
		startp = self.start
		endp = self.start + self.q
		state_vector_p = np.zeros((self.no_neurons+self.q+1))
		for i in range(self.no_neurons):
			state_vector_p[i] = current_state[i]
		sum = 0
		for j in range(startp,endp,1):
			if(previous_state[j] == 1):
				sum += 1
		sum2 = 0
		for i in range(self.q):
			if(sum<(i+1)):
				state_vector_p[i+self.no_neurons] = 0
			else:
				state_vector_p[i+self.no_neurons] = 1
				sum2 += 1
		if(sum2%2 == 0):
			state_vector_p[self.q+self.no_neurons] = 0
		else:
			state_vector_p[self.q+self.no_neurons] = 1
		return (state_vector_p[self.q+self.no_neurons])

	def determine_parity2(self):
		val = np.dot(self.Weights_p, self.state_vector_p) - self.threshold_p
		self.state_vector_p = np.copy(val[:self.no_neurons])
		return val[self.q + self.no_neurons]
			
	
network = Binary_Rec_Network(N,k,ita_STDP,ita_IP,q)
previous_state = np.zeros((N))                      # initially this is state -1 which would be required for finding state 1. At this moment all neurons are non-spiking i.e. none has fired
state_t = network.random_initial_state()             # u i.e. initial state i.e. state 0 
network.set_state_vector(state_t)
output = np.zeros((no_output_bits))
for i in range(output.shape[0]):
	next_state = network.update_function(previous_state,state_t)
	network.update_weights_antiSTDP(next_state,state_t)
	network.update_thresholds_IP(state_t)
	network.get_weightsp()
	network.get_thresholdsp()
	# Determine parity bit here
	bit = network.determine_parity(i+1,next_state,state_t)	
	print(bit)
	output[i] = bit	
	previous_state = state_t
	state_t = next_state
print (output)	
# make 100,000 training timesteps and then 8,000,000 testing timesteps
