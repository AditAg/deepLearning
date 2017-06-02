import numpy as np
import pandas as pd
import math
import random

hidden_layer_size = 10
no_classes = 10
K = 5                         # value of K for K-Nearest Neighbours
no_epochs = 5
alpha = 0.5                   # alpha is the learning parameter
lambda_val = 0.01
input_file = ".\Exploratory\emotion.csv"
y_label = 'Emotion'

#Implemented without regularization term
def clean_data(input_data):
    input_data = input_data.drop('Person Id',1)
    input_data = input_data.drop('Person SubID',1)
    return input_data
class Radial_Basis_Network:

    def __init__(self,input_layer_size,hidden_layer_size,output_size,lambda_val):
        self.indim = input_layer_size
        self.outdim = output_size
        self.no_centers = hidden_layer_size
        self.centers = [np.random.uniform(-1,1,self.indim) for i in range(self.no_centers)]
        self.W = np.random.normal(0,0.1,(self.outdim,self.no_centers+1))
        self.tolerance_level = 0.1
        self.scale_params = np.random.normal(0,0.1,self.no_centers)
        self.lambda_val = lambda_val

    def check_convergence(self,X,Y):
        #for i in range(len(X)):
        #   if((np.absolute(X[i]-Y[i])).all() > self.tolerance_level):
        #        return False
        #return True
        for i in range(len(X)):
            sum_vals = 0
            for j in range(X[i].shape[0]):
                sum_vals += (X[i][j]-Y[i][j])**2
            sum_vals = math.sqrt(sum_vals)
            if(sum_vals > self.tolerance_level):
                return False
        return True

    def distance(self,X,Y):
        dist = 0.0
        for i in range(X.shape[0]):
            dist +=(X[i]-Y[i])**2
        dist = math.sqrt(dist)
        return dist

    def clustering(self,train_data):
        indices = random.sample(range(0,train_data.shape[0]),self.no_centers)
        for i in range(self.no_centers):
            self.centers[i] = train_data[indices[i]]
        while(1):
            l={}
            for i in range(self.no_centers):
                l[i] = []
            for i in range(train_data.shape[0]):
                minimum_value = math.inf
                center = math.inf
                for j in range(self.no_centers):
                    dist = self.distance(self.centers[j],train_data[i])
                    if(dist < minimum_value):
                        minimum_value = dist
                        center = j
                    elif(dist == minimum_value):
                        if(train_data[i].all()==self.centers[j].all()):
                            center = j
                    else:
                        continue
                
                if(center in l.keys()):
                    l[center].append(i)
                else:
                    l[center] = []
                    l[center].append(i)
            new_centers = self.centers
            for i in range(self.no_centers):
                if(len(l[i])==0):
                    continue
                average_values = train_data[l[i][0]]
                for j in range(1,len(l[i])):
                    average_values += train_data[l[i][j]]
                average_values = average_values/(len(l[i]))
                new_centers[i] = average_values
            if(self.check_convergence(new_centers,self.centers)):
                break
            else:
                self.centers = new_centers

    def determine_scaling_parameter(self,K):
        for i in range(self.no_centers):
            r = 0
            l=[]
            for j in range(self.no_centers):
                if(i==j):
                    continue
                if(len(l)<K):
                    r= (self.distance(self.centers[i],self.centers[j]))**2
                    l.append(r)
                else:
                    r = (self.distance(self.centers[i],self.centers[j]))**2
                    if(r<max(l)):
                        l.remove(max(l))
                        l.append(r)
            val = math.sqrt(1.0*sum(l)/K)
            self.scale_params[i] = 1.0/val

    def gaussian_rbf(self,dist,index):
        z = dist*self.scale_params[index]
        z = z**2
        z = -1.0 * z
        z = math.exp(z)
        return z

    def derivative_gaussian_r(self,dist,index):
        z = self.gaussian_rbf(dist,index)
        z = -1.0*z
        z = 2.0*dist*self.scale_params[index]*self.scale_params[index]*z
        return z

    def derv2_phik_xij(self,data,k):
        sump=0
        for i in range(self.no_centers):
            val = -2.0*self.scale_params[i]*self.scale_params[i]
            val = val*self.W[k][i+1]*self.gaussian_rbf(self.distance(data,self.centers[i]),i)
            sump += val
        return sump

    def regularization_term(self,train_data):
        sump=0.0
        for i in range(self.outdim):
            print (i+1)
            for j in range(train_data.shape[0]):
                for k in range(train_data.shape[1]):
                    val = (self.derv2_phik_xij(train_data[j],i))**2
                    sump += val
        sump = (sump*self.lambda_val)/(2.0)
        return sump
            
    def prediction(self,data):
        hidden_layer = np.zeros((self.no_centers+1))
        for i in range(hidden_layer.shape[0]):
            if(i==0):
                hidden_layer[i] = 1
            else:
                hidden_layer[i] = self.gaussian_rbf(self.distance(data,self.centers[i-1]),i-1)
        output = np.zeros((self.outdim))
        for i in range(self.outdim):
            output[i]=0
            for j in range(hidden_layer.shape[0]):
                output[i] += hidden_layer[j]*self.W[i][j]
        return (hidden_layer,output)
        
    def Cost_function(self,train_data,y_values):
        cost_function = 0
        for i in range(train_data.shape[0]):
            difference = (y_values[i] - self.prediction(train_data[i])[1])
            difference = difference**2
            difference = sum(difference)
            cost_function += difference
        cost_function = (cost_function*1.0)/2
        #adding regularization term to cost function
        cost_function = cost_function + self.regularization_term(train_data)
        return cost_function

    def derivative_reg_term(self,data,j,k):
        sump = 4*self.lambda_val*data.shape[0]
        sump = sump*self.scale_params[k]*self.scale_params[k]
        sump = sump*self.gaussian_rbf(self.distance(data,self.centers[k]),k)
        value=0
        for i in range(self.no_centers):
            val = self.gaussian_rbf(self.distance(data,self.centers[i]),i)
            val = val*self.scale_params[i]*self.scale_params[i]
            val = val*self.W[j][i]
            value += val
        sump = sump*value
        return sump
    
    def linear_least_squares(self,train_data,y_values):
        hidden_layers = np.zeros((train_data.shape[0],self.no_centers+1))
        for i in range(hidden_layer.shape[0]):
            hidden_layer[i] = (self.prediction(train_data[i]))[0]
        mat = hidden_layers.transpose()
        mat2 = np.dot(mat,hidden_layers)
        mat2 = np.linalg.inv(mat2)
        mat2 = np.dot(mat2,mat)
        mat2 = np.dot(mat2,y_values)
        self.W = mat2.transpose()

    #We can also take normalized RBF architecture and then use operator projector training instead of SGD
    #Now output is the normalized output i.e. divide it by sum(RBF activation outputs) then see report for details
    def stochastic_gradient_descent(self,train_data,y_values,alpha,no_epochs):
        for p in range(no_epochs):
            print ("Epoch :",p+1)
            for i in range(train_data.shape[0]):
                print ("Training example :",i+1)
                for j in range(self.W.shape[0]):
                    for k in range(self.W.shape[1]):
                        value = sum(y_values[i] - self.prediction(train_data[i])[1])
                        if(k!=0):
                            value = value*self.gaussian_rbf(self.distance(train_data[i],self.centers[k-1]),k-1)
                            #derivative for regularization term
                            value = value - self.derivative_reg_term(train_data[i],j,k-1)
                        (self.W)[j][k] = (self.W)[j][k] + alpha*value

    def print_weights(self):
        print (self.W)
    def print_centers(self):
        print (self.centers)
    def print_scales(self):
        print (self.scale_params)
        
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
RBFNetwork = Radial_Basis_Network(train_data.shape[1],hidden_layer_size,no_classes,lambda_val)
print("Determining centers of RBF Activation Functions.....")
RBFNetwork.clustering(train_data)
print ("Determined centers......")
RBFNetwork.print_centers()
print("Determining scaling parameters of RBF Activation Functions....")
RBFNetwork.determine_scaling_parameter(K)
print ("Determined scaling parameters.....")
RBFNetwork.print_scales()
print (RBFNetwork.Cost_function(train_data,y_values))
print("Determining weights....")
RBFNetwork.stochastic_gradient_descent(train_data,y_values,alpha,no_epochs)
print("Determined Weights ......")
print (RBFNetwork.Cost_function(train_data,y_values))
#RBFNetwork.print_weights()
# test for output for any test_data
# (hidden_layer,output) = RBFNetwork.prediction(test_data)
