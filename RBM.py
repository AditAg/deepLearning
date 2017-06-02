import numpy as np
import pandas as pd
import math
import random
from itertools import product

input_file = ".\Exploratory\emotion.csv"
y_label = 'Emotion'
learning_rate = 0.1
K = 1.3806485 *math.pow(10,23)          # Boltzmann Constant                  

def clean_data(input_data):
    input_data = input_data.drop('Person Id',1)
    input_data = input_data.drop('Person SubID',1)
    return input_data

# function to make all possible state vectors 
def make_truth_assignments(length):
    truth_assignments=[]
    n=2**(length)
    ta=[1 for j in range(length)]
    for i in range(n):
        truth_assignments.append([])
        j=1
        while(j<n):
            if(i%j==0):
                ta[int(math.log(j,2))]=(ta[int(math.log(j,2))]+1)%2
            j=j*2
        for j in range(length-1,-1,-1):
            truth_assignments[i].append(ta[j])
    return truth_assignments

class RBM:

    def __init__(self,no_hidden_neurons,no_input_neurons,input_data,learning_rate):
        self.no_hidden_neurons = no_hidden_neurons
        self.input_data = input_data     #These are the input values of neurons
        self.no_visible_neurons = no_input_neurons
        self.learning_rate = learning_rate
        self.Weights = np.random.normal(0,0.1,(self.no_hidden_neurons,self.no_visible_neurons))
        self.input_biases = np.random.normal(0,0.1,(self.no_visible_neurons,1))
        self.hidden_biases = np.random.normal(0,0.1,(self.no_hidden_neurons,1))
    
    def network_energy(self,x,h):
        energy = 0
        for i in range(self.Weights.shape[0]):
            for j in range(self.Weights.shape[1]):
                energy += (self.Weights[i][j]*h[i]*x[j])
        for i in range(x.shape[0]):
            energy += (self.input_biases[i][0]*x[i])
        for i in range(h.shape[0]):
            energy += (self.hidden_biases[i][0]*h[i])
        energy = -1.0 * energy
        return energy

    def sigmoid(self,x):
        return 1.0/(1.0+math.exp(-x))

    def partition_function(self):
        val1 = self.no_hidden_neurons
        val2 = self.no_visible_neurons
        l1 = list(product(range(2),val1))
        l2 = list(product(range(2),val2))
        sum1 = 0
        for i in range(len(l2)):
            z = np.array(l2[i])
            for j in range(len(l1)):
                z1 = np.array(l1[j])
                energy = self.network_energy(z,z1)
                energy = math.exp(-1.0*energy)
                sum1 += energy
        return sum1

    def prob_input(self,x,h):
        energy = math.exp(-1.0*self.network_energy(x,h))
        prob = (energy*1.0)/(self.partition_function)
        return prob
    # probability of p(hj |x) hj=0 and hj=1
    def prob_hx(self,index,x):
        sum1 = 0
        for i in range(x.shape[0]):
            sum1 += (self.Weights[index][i]*x[i])
        sum1 += (self.hidden_biases[index][0])
        return (self.sigmoid(-1.0*sum1),self.sigmoid(sum1))
    
    #probability of p(xj|h) xj=0 and xj=1
    def prob_xh(self,index,h):
        sum1 = 0
        for i in range(h.shape[0]):
            sum1 += (self.Weights[i][index]*h[i])
        sum1 += (self.input_biases[index][0])
        return (self.sigmoid(-1.0*sum1),self.sigmoid(sum1))

    def prob_x(self,x):
        sum1 = (np.dot(self.input_biases.transpose(),x))[0][0]
        for j in range(self.no_hidden_neurons):
            val = self.hidden_biases[j][0]
            for k in range(x.shape[0]):
                val += self.Weights[j][k]*x[k]
            val = math.exp(val)
            sum1 += math.log(1+val)
        return (math.exp(sum1*1.0))/(self.partition_function)

    def train_data(self,no_epochs):
        hidden_layer = np.random.randint(2,size =(self.no_hidden_neurons))
        for r in range(no_epochs):
            for i in range(self.input_data.shape[0]):
                #Positive Phase
                probs1 = list()
                for j in range(self.no_hidden_neurons):
                    prob = self.prob_hx(j,self.input_data[i])[1]
                    probs1.append(prob)
                    if(prob > np.random.rand()):    # should it be 0.5
                        hidden_layer[j] = 1
                    else:
                        hidden_layer[j] = 0
                #Negative Phase ---- using only 1 step of Gibbs sampling 
                probs2 = list()
                z = list()
                for j in range(self.no_visible_neurons):
                    prob = self.prob_xh(j,hidden_layer)[1]
                    probs2.append(prob)
                    if(prob > np.random.rand()):     #should it be 0.5
                        #self.input_data[i][j] = 1    # use this if you want to change the input data as well
                        z.append(1)
                    else:
                        #self.input_data[i][j] = 0
                        z.append(0)
                probs3 = list()
                z = np.array(z)
                for j in range(self.no_hidden_neurons):
                    #prob = self.prob_hx(j,self.input_data[i])[1]
                    prob = self.prob_hx(j,z)[1]
                    probs3.append(prob)
                    if(prob > np.random.rand()):    # should it be 0.5
                        hidden_layer[j] = 1
                    else:
                        hidden_layer[j] = 0
                #Updating the weights
                for j in range(self.Weights.shape[0]):
                    for k in range(self.Weights.shape[1]):
                        val = probs1[j] - (probs2[k]*probs3[j])
                        self.Weights[j][k] += self.learning_rate*val

    # Make function for K steps of gibbs sampling
    def Gibbs_sampling(self,data,K):
        hidden_layer = np.random.randint(2,size =(self.no_hidden_neurons))
        for r in range(K):
            for i in range(data.shape[0]):
                #Positive Phase
                for j in range(self.no_hidden_neurons):
                    prob = self.prob_hx(j,data)[1]
                    if(prob > np.random.rand()):    # should it be 0.5
                        hidden_layer[j] = 1
                    else:
                        hidden_layer[j] = 0
                for j in range(self.no_visible_neurons):
                    prob = self.prob_xh(j,hidden_layer)[1]
                    if(prob > np.random.rand()):     #should it be 0.5
                        data[j] = 1
                    else:
                        data[j] = 0
        return data

    def contrastive_divergence(self,data,K):
        for i in range(data.shape[0]):
            negative_sample = self.Gibbs_sampling(data[i],K)
            h_data = np.zeros((self.no_hidden_neurons,1))
            h_neg_sample = np.zeros((self.no_hidden_neurons,1))
            for j in range(self.no_hidden_neurons):
                prob1 = self.prob_hx(j,data[i])[1]
                prob2 = self.prob_hx(j,negative_sample)[1]
                h_data[j][0] = prob1
                h_neg_sample[j][0] = prob2
            new_data = np.reshape(data[i],(data[i].shape[0],1))
            negative_sample = np.reshape(negative_sample,(negative_sample.shape[0],1))
            z1 = np.dot(h_data,new_data.transpose())
            z2 = np.dot(h_neg_sample,negative_sample.transpose())
            self.Weights = self.Weights + self.learning_rate*(z1 - z2)
    
    def print_weights(self):
        print (self.Weights)

    def run_visible(self,data):
        num_examples = data.shape[0]
        hidden_units = np.random.randint(2,size = (num_examples,self.no_hidden_neurons))
        for i in range(num_examples):
            for j in range(self.no_hidden_neurons):
                prob = self.prob_hx(j,data[i])[1]
                if(prob > np.random.rand()):    # should it be 0.5
                    hidden_units[i][j] = 1
                else:
                    hidden_units[i][j] = 0
        return hidden_units

    def run_hidden(self,data):
        num_examples = data.shape[0]
        visible_units = np.random.randint(2,size = (num_examples,self.no_visible_neurons))
        for i in range(num_examples):
            for j in range(self.no_visible_neurons):
                prob = self.prob_xh(j,data[i])[1]
                if(prob > np.random.rand()):    # should it be 0.5
                    visible_units[i][j] = 1
                else:
                    visible_units[i][j] = 0
        return visible_units

input_data = pd.read_csv(input_file)
# clean data to make all features as numbers    
input_data = clean_data(input_data)
input_data = input_data.drop(y_label,1)
train_data = input_data.as_matrix()
train_data = np.random.randint(2,size = (10,6))
HN = RBM(3,train_data.shape[1],train_data,0.01)
print("Hello")
HN.print_weights()
HN.train_data(50)
print("Bye")
HN.print_weights()

test_data = np.random.randint(2,size = (4,train_data.shape[1]))
print (test_data)
hidden_data = HN.run_visible(test_data)
output = HN.run_hidden(hidden_data)
print (output)
