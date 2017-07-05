import numpy as np
import random
import scipy.linalg
import math
import cv2
from matplotlib import pyplot as plt

height,width = 40,50
train_len = int(height*width)
size_binary_vector = 256
leak_rate = 0.07
reg_constant = 0.5
chunk_size = 200
no_sequences = chunk_size
no_reservoir_neurons = 0.95 * chunk_size

def convert_image(image):
	image1 = cv2.resize(image,(width,height))
	image = np.ndarray.flatten(image1)
	data = np.zeros((size_binary_vector,image.shape[0]))
	for i in range(image.shape[0]):
		data[(image[i])%size_binary_vector][i] = 1
	return (data,image1)

def convert_to_image(data):
	img = np.zeros((data.shape[1]),dtype = int)
	for i in range(data.shape[1]):
		index = np.argmax(data[:,i])
		img[i] = int(index)
	print(img)
	img = np.reshape(img,(height,width))
	return img

def add_noise(x):                                                            # define different methods to add noise to distort pixel here
	return (256-x)%256

def random_distort_image(image,percent):                                            # percent% of the pixels randomly distorted plaintext sensitivity
	img = np.ndarray.flatten(image)
	no_pixels = img.shape[0]
	no_pix_distort = (percent/100.0)*no_pixels
	pixels = random.sample(range(0,no_pixels),int(no_pix_distort))
	for i in range(len(pixels)):
		img[pixels[i]] = add_noise(img[i])
	img = np.reshape(img,(image.shape))
	return img	
		
def get_data():                                                                   # Here get the input sequences
	#return np.random.randint(0,2,(size_binary_vector,train_len))
	data = np.zeros((size_binary_vector,train_len))
	for i in range(train_len):
		#d = np.random.choice([0,1],size=(size_binary_vector,),p=[((size_binary_vector-1)/size_binary_vector),(1/size_binary_vector)])
		#data[:,i] = np.copy(d)
		index = np.random.randint(size_binary_vector)
		data[index][i] = 1
	return data	

def get_dummy_node():                                                       #Here we obtain the dummy node b0 so that our recall is independent of it.
	#return np.ones((size_binary_vector,1))
	return (np.random.choice([0,1],size=(size_binary_vector,1),p=[((size_binary_vector-1)/size_binary_vector),(1/size_binary_vector)]))	

class Echo_State_Network(object):
	
	def __init__(self,no_input_neurons,no_reservoir_neurons,no_output_neurons,leaking_rate,regularization_constant):
		self.indim = int(no_input_neurons)
		self.hdim = int(no_reservoir_neurons)
		self.outdim = int(no_output_neurons)
		self.leaking_rate = leaking_rate
		self.input_weights = (np.random.rand(1+self.indim,self.hdim) - 0.5) * 2.0
		self.hidden_weights = (np.random.rand(self.hdim,self.hdim) - 0.5) * 2.0
		self.output_weights = (np.random.rand(self.hdim,self.outdim) - 0.5) * 2.0
		self.input_data = train_data	
		self.reg_const = regularization_constant	
	
	def determine_spectral_radius(self):
		rhoW = max(abs(np.linalg.eig(self.hidden_weights)[0]))
		self.hidden_weights = self.hidden_weights * (1.25/rhoW)
	
	#def determine_scale(self):                                        # scale a of self.input_weights

	def sigmoid(self,x):
		return 1.0/(1.0 + np.exp(-1.0 * x))
	
	def fo(self,x):                                                    # can use linear or softmax function as well.
		return np.tanh(x)
	
	def softmax(self,x):
		e_x = np.exp(x - np.max(x))
		return (e_x/ e_x.sum())
	
	def predict_output(self,input_data,hidden_output):                 #input_data is a numpy array of size self.indim x 1 and hidden_output is a numpy array of size self.hdim x 1
		val = np.dot(self.hidden_weights.T,hidden_output)
		val = val + np.dot(self.input_weights.T , np.vstack((1,input_data)))
		val = self.leaking_rate * self.sigmoid(val)
		hidden_layer = (1 - self.leaking_rate)*hidden_output + val
		#output_layer = self.fo(np.dot(self.output_weights.T, hidden_layer))
		output_layer = self.softmax(np.dot(self.output_weights.T, hidden_layer))		
		return (hidden_layer,output_layer)		
		
	def generate_reservoir(self,X):
		R = np.zeros((self.hdim,X.shape[1]))
		hidden_layer = np.zeros((self.hdim,1))
		for t in range(X.shape[1]):
			x_t = np.reshape(X[:,t],(self.indim,1))
			hidden_layer,_ = self.predict_output(x_t,hidden_layer)
			R[:,t] = np.copy(np.reshape(hidden_layer,(1,self.hdim)))
		return R		

	def train_out_matrix(self,X,Y):                                    # X is a training sequence of input data of shape self.indim X T and Y is the desired output sequence of shape self.outdim x T
		R = self.generate_reservoir(X)				   # R is a numpy array of shape self.hdim x T
		val = np.dot(R,R.T) + (self.reg_const * np.identity(self.hdim))
		val = np.linalg.inv(val)
		self.output_weights = (np.dot(Y,np.dot(R.T,val))).T
	
	#def softmax(self,X):
	#	y = np.atleast_2d(X)
	#	axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
	#	y = y - np.expand_dims(np.max(y, axis = axis), axis)
	#	y = np.exp(y)
	#	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
	#	p = y / ax_sum
	#	if len(X.shape) == 1: p = p.flatten()
	#	return p
	
	def predict_message_Bob(self,input_x0,size):                             # Here we pass the dummy node as the initial input
		Y = np.zeros((self.outdim,size))
		inp = np.copy(input_x0)
		hidden_layer = np.zeros((self.hdim,1))
		for i in range(size):
			(hidden_layer,output_layer) = self.predict_output(inp,hidden_layer)
			#y_output = self.softmax(output_layer)
			#Binarize the output
			y_output = np.copy(output_layer)		
			index = np.argmax(y_output,axis = 0)
			y_output = np.zeros((y_output.shape))
			y_output[index] = 1
			Y[:,i] = np.copy(np.reshape(y_output,self.outdim))
			inp = y_output
		return Y
	def get_output_weights(self):
		return self.output_weights
	
	def set_output_weights(self,weights):
		self.output_weights = weights

image = cv2.imread('lena.jpg',0)	
#train_data = get_data()
(train_data,resized_image) = convert_image(image)
#print(resized_image)
#print(train_data[:,0],train_data[:,1],train_data.shape)
dummy_node = get_dummy_node()
#train_data = np.hstack((dummy_node,train_data))
#Y_actual = train_data[:,1:]
#X_actual = train_data[:,:-1]
#network = Echo_State_Network(X_actual.shape[0],no_reservoir_neurons,Y_actual.shape[0],leak_rate,reg_constant)
network = Echo_State_Network(train_data.shape[0],no_reservoir_neurons,train_data.shape[0],leak_rate,reg_constant)
network.determine_spectral_radius()

#Bob's output
Bob_prediction = np.zeros((train_data.shape))
for i in range(math.ceil(train_len/no_sequences)):
	end = (i+1)*no_sequences
	if(end > train_len):
		end = train_len
	#X = X_actual[:,i*no_sequences:end]
	#Y = Y_actual[:,i*no_sequences:end]
	Y = train_data[:,i*no_sequences:end]	
	X = train_data[:,i*no_sequences:end-1]
	X = np.hstack((dummy_node,X))
	network.train_out_matrix(X,Y)

	# Bob's predicted message
	Bob_output = network.predict_message_Bob(dummy_node,end - (i*no_sequences))
	Bob_prediction[:,i*no_sequences:end] = np.copy(Bob_output)
print (resized_image)
predicted_image = convert_to_image(Bob_prediction)
print (resized_image.shape,predicted_image.shape)
output_images = np.concatenate((resized_image,predicted_image),axis=1)
print(predicted_image.shape,predicted_image)
#output_image = cv2.resize(predicted_image,(image.shape[0],image.shape[1]))
#print(predicted_image)
#cv2.imshow('frame',resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
dpi = 72.
xinch = height / dpi
yinch = width / dpi
fig = plt.figure(figsize = (xinch,yinch))
a = fig.add_subplot(1,2,1)
plt.imshow(resized_image,interpolation = 'nearest',cmap='gray')
a =fig.add_subplot(1,2,2)
plt.imshow(predicted_image,cmap='gray')
plt.show()

