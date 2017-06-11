#try that output is a 5Xstep_size matrix and not a 5X1 output 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Input,Flatten
from keras.optimizers import SGD
from keras import backend as K
from keras.preprocessing import sequence
import random
from sklearn.metrics import precision_recall_fscore_support

m = 1930                             # 300 from DoS,R2L,Probe attacks each, 30 from U2R attack and 1000 from normal instances
hidden_layer_size = 80
learning_rate = 0.01
time_step_size = 100
batch_size = 50
no_epochs = 20

class Intrusion_detection:
	
	def __init__(self,path_to_data,hidden_layer_size,learning_rate,time_step_size,batch_size,no_epochs):
		self.no_labels = 5
		self.dict_attacks = {'DoS':['back','land','neptune','pod','smurf','teardrop'],'R2L':['ftp-write','guess-passwd','imap','multihop','phf','spy', 'warezclient','warezmaster'],'U2R':['buffer_overflow','loadmodule','perl','rootkit'],'Probe':['ipsweep','nmap','portsweep','satan'],'normal':['normal']}
		self.hidden_layer_size = hidden_layer_size
		self.learning_rate = learning_rate
		self.output_shape = self.no_labels
		self.time_step_size = time_step_size
		self.batch_size = batch_size
		self.no_epochs = no_epochs
		self.no_test_datasets = 10
		self.test_size = 5000
		(self.train_data,self.labels,self.no_features,self.test_data,self.test_labels) = self.get_data(path_to_data)	
		self.input_size = self.no_features

	def get_data(self,path_to_data):
		file = open(path_to_data,'r')
		X = file.read()
		file.close()
		Z = X.strip().split('\n')
		count1,count2,count3,count4,count5,count6 = 0,0,0,0,0,0
		no_features = len(Z[0].strip().split(','))-1
		train_data = np.zeros((m,no_features))
		labels = np.zeros((m,self.no_labels))
		test_data=[]
		test_values=[]
		k=0
		z  = list()
		z2 = list()
		z3 = list()
		for i in Z:
			p = i.strip().split(',')
			if(p[1] not in z):
				z.append(p[1])
			if(p[2] not in z2):
				z2.append(p[2])
			if(p[3] not in z3):
				z3.append(p[3])
		for i in range(len(Z)):
			r = Z[i].strip().split(',')
			features = r[0:-1]
			features[1] = [i for i,x in enumerate(z) if x == features[1]][0]
			features[2] = [i for i,x in enumerate(z2) if x == features[2]][0]
			features[3] = [i for i,x in enumerate(z3) if x == features[3]][0] 
			label = r[-1][:-1]
			if(label in self.dict_attacks['DoS'] and count1<300):
				train_data[k] = np.array(features).astype(float)                       # can use np.chararray to store strings here 
				labels[k][0] = 1
				k = k+1
				count1 = count1 +1
			elif(label in self.dict_attacks['R2L'] and count2<300):
				train_data[k] = np.array(features).astype(float)
				labels[k][1] = 1
				k = k+1
				count2 = count2 +1
			elif(label in self.dict_attacks['U2R'] and count3<30):
				train_data[k] = np.array(features).astype(float)
				labels[k][2] = 1
				k = k+1
				count3 = count3 +1
			elif(label in self.dict_attacks['Probe'] and count4<300):
				train_data[k] = np.array(features).astype(float)
				labels[k][3] = 1
				k = k+1
				count4 = count4 +1
			elif(label in self.dict_attacks['normal'] and count5<1000):
				train_data[k] = np.array(features).astype(float)
				labels[k][4] = 1
				k = k+ 1
				count5 = count5 +1
			elif(count6<30000):
				test_data.append(np.array(features).astype(float))
				yval = np.zeros((5))
				if(label in self.dict_attacks['DoS']):
					yval[0] = 1
				elif(label in self.dict_attacks['R2L']):
					yval[1] = 1
				elif(label in self.dict_attacks['U2R']):
					yval[2] = 1
				elif(label in self.dict_attacks['Probe']):
					yval[3] = 1
				else:
					yval[4] = 1
				test_values.append(yval)
				count6 = count6+1
			else:
				continue
		test_vals = np.zeros((self.no_test_datasets,self.test_size,no_features))
		test_y = np.zeros((self.no_test_datasets,self.test_size,self.no_labels))
		for i in range(test_vals.shape[0]):
			index_vals = random.sample(range(0,len(test_data)),self.test_size)
			for j in range(len(index_vals)):
				test_vals[i][j] = test_data[index_vals[j]]
				test_y[i][j] = test_values[index_vals[j]]
		del Z
		del X,z,z2,z3
		return (train_data,labels,no_features,test_vals,test_y)
	
	def normalize_data(self):
		scaler = MinMaxScaler(feature_range =(0,1))
		self.train_data = scaler.fit_transform(self.train_data)

	def reshape_data(self):
		dataX=[]
		data_test=[]
		for i in range(self.train_data.shape[0]-self.time_step_size+1):              #time_step_size should have no use here since we arent doing time series prediction or sequence prediction we are simply doing classification and here for each instance we have an output not that the next object in sequence is the output for current time_step elements as is generally the case in sequences.(E.g.- predicting next character in sequence)
			a = self.train_data[i:i+self.time_step_size,:]
			dataX.append(a)
		for j in range(self.test_data.shape[0]):
			data_test2 = []
			for i in range(self.test_data.shape[1]-self.time_step_size+1):   
				a = self.test_data[j,i:i+self.time_step_size,:]
				data_test2.append(a)
			data_test.append(data_test2)
		#dataX = list(self.train_data)
		#data = sequence.pad_sequences(dataX,maxlen=self.time_step_size)		
		self.train_data = np.array(dataX)
		self.train_data = np.reshape(self.train_data,(self.train_data.shape[0],self.time_step_size,self.train_data.shape[2]))
		self.labels = self.labels[self.time_step_size-1:,:]
		self.test_data = np.array(data_test)
		self.test_labels = self.test_labels[:,self.time_step_size-1:,:]

	def precision(self,y_true, y_pred):   #Here it assumes only binary classification to do else predicted_positives etc. should be changed
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision

	def recall(self,y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def fallout(self,y_true,y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		false_positives = predicted_positives - true_positives
		false_negatives = possible_positives - true_positives
		#true_negatives = 
		#return false_positives/(false_positives+true_negatives+K.epsilon())

	def build_network(self):
		model=Sequential()
		model.add(LSTM(self.hidden_layer_size, input_shape = (self.time_step_size,self.input_size),kernel_initializer = 'glorot_normal',recurrent_initializer='orthogonal'))
		#model.add(LSTM(self.hidden_layer_size,input_shape=(self.input_size,)))   # LSTM without any time_step_size		
		#model.add(Flatten())
		model.add(Dense(self.output_shape,activation = 'softmax'))
		sgd = SGD(lr=self.learning_rate,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(optimizer = sgd,loss='mean_squared_error',metrics=['accuracy',self.precision,self.recall])    
		# Detection Rate(DR): TP/(TP + FN)         (TPR or Sensitivity or Recall)
		# False Alarm Rate(FAR) : FP/(TN + FP)     (FPR or Fall-out)
		# Efficiency : DR/FAR                      (Positive Likelihood Ratio)
		self.model = model
		
	def train_on_data(self):
		self.model.fit(self.train_data,self.labels,batch_size = self.batch_size,epochs = self.no_epochs)
	
	def test_on_data(self):
		vals = {0:'DoS',1:'R2L',2:'U2R',3:'Probe',4:'normal'}
		for i in range(self.test_data.shape[0]):
			#score = self.model.evaluate(self.test_data[i],self.test_labels[i],batch_size = 200)
			#print('Test score:',score[0])
			#print('Test accuracy:',score[1])
			output = self.model.predict(self.test_data[i])
			yvals = []
			yvals2 =[]
			for j in range(output.shape[0]):
				index = np.argmax(output[j])
				index2 = np.argmax(self.test_labels[i][j])
				yvals.append(vals[index])
				yvals2.append(vals[index2])
			yvals = np.array(yvals)
			yvals2 = np.array(yvals2)
			print("Test set ",i+1,precision_recall_fscore_support(yvals2,yvals,average=None))
				
			
	def print_data(self):
		print(self.train_data)
		print(self.labels)

network = Intrusion_detection('kddcup.data.corrected',hidden_layer_size,learning_rate,time_step_size,batch_size,no_epochs)
network.normalize_data()
network.reshape_data()
network.build_network()
network.train_on_data()
network.test_on_data()
#network.print_data()
