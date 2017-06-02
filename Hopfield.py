import numpy as np
import pandas as pd
import math
import random

hidden_layer_size = 10
no_classes = 10
K = 5                         # value of K for K-Nearest Neighbours
no_epochs = 5
alpha = 0.5                   # alpha is the learning parameter
input_file = ".\Exploratory\emotion.csv"
y_label = 'Emotion'

def clean_data(input_data):
    input_data = input_data.drop('Person Id',1)
    input_data = input_data.drop('Person SubID',1)
    return input_data

class Hopfield_Network:

    def __init__(self,no_neurons,input_data,mark):
        self.no_neurons = no_neurons
        #self.Weights = np.random.normal(0,0.1,(self.no_neurons,self.no_neurons))
        self.thresholds = np.random.uniform(0.2,0.7,self.no_neurons)
        #for i in range(self.Weights.shape[0]):
        #    self.Weights[i][i] = 0
        #    for j in range(i):
        #        self.Weights[i][j]=self.Weights[j][i]
        self.input_data = input_data     #These are the input values of neurons
        self.Weights = np.zeros((self.no_neurons,self.no_neurons))
        self.learning_rate = (1.0/self.input_data.shape[0])
        self.mark = mark                  #mark indicates whether input is {-1,1} or {0,1}
        self.stored_energies = np.zeros((self.input_data.shape[0]))

    def network_energy(self,data):
        energy = 0
        for i in range(self.no_neurons):
            energy += (self.thresholds[i]*data[i])
            for j in range(self.no_neurons):
                value = (self.Weights[i][j]*data[i]*data[j])
                value = -1.0*value
                energy += (value/2.0)
        return energy

    def train_input_data(self):
        if(self.mark == 1):
            identity = np.eye(self.no_neurons,dtype=int)
            for i in range(self.input_data.shape[0]):
                t1 = self.input_data[i].reshape((self.input_data[i].shape[0],1))
                t2 = self.input_data[i].reshape((1,self.input_data[i].shape[0]))
                t = np.dot(t1,t2)
                value = self.learning_rate*(t-identity)
                self.Weights = self.Weights + value
        elif(self.mark == 2):
            for l in range(self.input_data.shape[0]):
                for i in range(self.Weights.shape[0]):
                    for j in range(self.Weights.shape[1]):
                        value = ((2*self.input_data[l][i]) -1)*((2*self.input_data[l][j] - 1))
                        self.Weights[i][j] = self.Weights[i][j] + value
                        
    def store_energies(self):
        for i in range(self.input_data.shape[0]):
            self.stored_energies[i] = self.network_energy(self.input_data[i])
        
    def train_new(self,data):
        if(self.mark == 1):
            for i in range(self.Weights.shape[0]):
                for j in range(self.Weights.shape[1]):
                    if(i==j):
                        continue
                    self.Weights[i][j] = self.Weights[i][j] + (self.learning_rate*data[i]*data[j])
        else:
            for i in range(self.Weights.shape[0]):
                for j in range(self.Weights.shape[1]):
                    value = ((2*data[i]) -1)*((2*data[j] - 1))
                    self.Weights[i][j] = self.Weights[i][j] + value
        energy = self.network_energy(data)
        self.stored_energies = np.insert(self.stored_energies,0,energy)
        self.input_data = np.insert(self.input_data,0,data,axis = 0)

    def funct(self,x):
        if(x>=0):
            return 1
        else:
            return -1

    def update_neuron(self,index,data):
        sum_weights=0
        for i in range(self.Weights[index].shape[0]):
            sum_weights += (self.Weights[index][i]*data[i])
        sum_weights = sum_weights - self.thresholds[index]
        data[index] = self.funct(sum_weights)
        return data[index]

    def test_data(self,data):
        indices = random.sample(range(0,data.shape[0]),data.shape[0])
        i=0
        old_data = data
        while(1):
            energy = self.network_energy(data)
            for i in range(self.stored_energies.shape[0]):
                if(energy == self.stored_energies[i]):
                    return self.input_data[i]
            index = indices[i]
            i = ((i+1)%data.shape[0])
            data[index] = self.update_neuron(index,data)
            if(i==0):
                a=0
                for i in range(data.shape[0]):
                    if(data[i]!=old_data[i]):
                        a=1
                        break
                if(a==0):
                    #It has converged to a local minima
                    return data
                else:
                    old_data = data
        
input_data = pd.read_csv(input_file)
input_data.insert(0,'x0',np.random.normal(input_data.shape[0]))
input_data = clean_data(input_data)
# clean data to make all features as numbers    
#output y should be a Mxm matrix where M= no of training examples and m=dimension of output
y = input_data[y_label]
y_values = np.zeros((y.shape[0],no_classes))
for i in range(y_values.shape[0]):
    y_values[i][y[i]-1] = 1

input_data = input_data.drop(y_label,1)
train_data = input_data.as_matrix()
