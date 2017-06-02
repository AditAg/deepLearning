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

class Hopfield_Network:

    def __init__(self,no_hidden_neurons,no_input_neurons,input_data,learning_rate):
        self.no_hidden_neurons = no_hidden_neurons
        self.input_data = input_data     #These are the input values of neurons
        self.no_visible_neurons = no_input_neurons
        self.learning_rate = learning_rate
        self.no_neurons = self.no_hidden_neurons + self.no_visible_neurons
        #self.W = np.random.normal(0,0.1,(self.no_visible_neurons,self.no_hidden_neurons))
        #self.L = np.random.normal(0,0.1,(self.no_visible_neurons,self.no_visible_neurons))
        #self.J = np.random.normal(0,0.1,(self.no_hidden_neurons,self.no_hidden_neurons))
        self.Weights = np.random.normal(0,0.1,(self.no_neurons,self.no_neurons))
        self.Weights = (self.Weights + (self.Weights.transpose()))/2
        for i in range(self.no_neurons):
            self.Weights[i][i] = 0.0
        self.thresholds = np.random.normal(0,0.1,(self.no_neurons))
        self.T = 1.0
        self.stored_energies = np.zeros((2**self.no_neurons))
        self.tolerance_level = 0
    
    #E(V) , considering W,L and J as the network parameters and no thresholds   
    #def network_energy(self,visible_data,hidden_data):
    #    energy = 0
    #    v = visible_data.reshape((visible_data.shape[0],1))
    #    h = hidden_data.reshape((hidden_data.reshape[0],1))
    #    val1 = np.dot(self.L,v)
    #    val1 = np.dot(v.transpose(),val1)
    #    val1 = (-1.0/2)*val1
    #    val2 = np.dot(self.J,h)
    #    val2 = np.dot(h.transpose(),val2)
    #    val2 = (-1.0/2)*val2
    #    val3 = np.dot(self.W,h)
    #    val3 = np.dot(v.transpose(),val2)
    #    val3 = (-1.0) * val3
    #    energy = val1+val2+val3
    #    return energy

    def network_energy(self,state_vector):
        energy = 0
        for i in range(state_vector.shape[0]):
            for j in range(i+1,state_vector.shape[0]):
                energy += (self.Weights[i][j]*state_vector[i]*state_vector[j])
        energy = -1.0 * energy
        for i in range(state_vector.shape[0]):
            energy += (self.thresholds[i]*state_vector[i])
        return energy

    def all_energies(self):
        val = self.no_neurons
        l = list(product(range(2),repeat = val))
        for i in range(len(l)):
            z = np.array(l[i])
            self.stored_energies[i] = self.network_energy(z)
            
    #delta(E_i) = E_{si=0} - E_{si=1}
    def delta_Ei(self,state_vector,index):
        diff_energy = 0
        for i in range(0,index):
            diff_energy += self.Weights[i][index]*state_vector[i]
        for i in range(index+1,state_vector.shape[0]):
                diff_energy += self.Weights[index][i]*state_vector[i]
        diff_energy += self.thresholds[index]
        return diff_energy

    def sigmoid(self,x):
        return 1.0/(1.0+math.exp(-x))

    def prob_unit_1(self,state_vector,index):
        val = self.delta_Ei(state_vector,index)
        z = K*self.T
        val = val/z
        return self.sigmoid(val)

    def probability_sv(self,state_vector):
        energy = (math.exp(-1.0*self.network_energy(state_vector)))
        sum1 = 0
        for i in range(self.stored_energies.shape[0]):
            val = math.exp(-1.0*self.stored_energies[i])
            sum1 += val
        return (energy/sum1)

    #function to find probabilty of input vector without any hidden values
    def prob_visiblesv(self,visible_data):
        val = self.no_hidden_neurons
        l = list(product(range(2),repeat = val))
        sum1 = 0
        for i in range(len(l)):
            z = np.array(l[i])
            state_vector = np.concatenate((visible_data,z),axis=0)
            sum1 += self.probability_sv(state_vector)
        return sum1

    def train_data(self):
        for m in range(self.input_data.shape[0]):
            for i in range(self.Weights.shape[0]):
                for j in range(self.Weights.shape[1]):
                    sum1 = 0
                    sum2 = 0
                    val1 =self.no_hidden_neurons
                    val2 =self.no_neurons
                    l1 = list(product(range(2),repeat = val1))
                    l2 = list(product(range(2),repeat = val2))
                    total_energy = 0
                    for k in range(len(l1)):
                        z = np.array(l1[k])
                        state_vector = np.concatenate((self.input_data[m],z),axis=0)
                        energy = (math.exp(-1.0*self.network_energy(state_vector)))
                        total_energy += energy
                        sum1 += (energy*state_vector[i]*state_vector[j])
                    sum1 = (sum1/total_energy)
                    for k in range(len(l2)):
                        state_vector = np.array(l2[k])
                        value = (self.probability_sv(state_vector) * state_vector[i]*state_vector[j])
                        sum2 += value
                    sum2 = -1.0 *sum2
                    self.Weights[i][j] = self.Weights[i][j] + (self.learning_rate*(sum1+sum2))

    def update_value(self,index,state_vector):
        if(self.prob_unit_1(state_vector,index)> 0.5):
            return 1
        else:
            return 0
            
        
    def test(self,visible_data):
        #initialize hidden_units randomly
        hidden_units = np.random.randint(2,size = self.no_hidden_neurons)
        state_vector = np.concatenate((visible_data,hidden_units),axis=0)
        #initialize T to some high value
        old_energy = self.network_energy(state_vector)
        while(1):
            #choosing a random hidden_unit and update it
            ##z = random.randrange(self.no_visible_neurons,self.no_neurons)
            z = random.randrange(0,self.no_neurons)
            state_vector[z] = self.update_value(z,state_vector)
            #reduce T according to some annealing procedure
            new_energy = self.network_energy(state_vector)
            #convergence criterion i.e. "Thermal Equilibrium"
            if((new_energy - old_energy)>=self.tolerance_level):
                break
            else:
                old_energy = new_energy
        output = state_vector[:self.no_visible_neurons]
        return output
    
input_data = pd.read_csv(input_file)
# clean data to make all features as numbers    
input_data = clean_data(input_data)
input_data = input_data.drop(y_label,1)
train_data = input_data.as_matrix()
train_data = np.random.randint(2,size = (10,6))
HN = Hopfield_Network(3,train_data.shape[1],train_data,0.01)
print("Hello")
HN.all_energies()
print("Bye")
HN.train_data()
test_data = np.random.randint(2,size = train_data.shape[1])
print (test_data)
print (HN.test(test_data))
