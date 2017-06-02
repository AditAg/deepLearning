import cv2
import numpy as np
import math

filter_size =(5,5,3)
stride_length=(1,1)
no_filters=3
padding = 0
pool_size =(2,2)
pool_type="max"
no_classes=10
learning_rate = 10

class ConvolutionalNN(object):

    def sigma(self,x):
        return 1.0/(1.0+math.exp(-x))

    def flatten(self,X):
        return X.flatten('F')
    
    def padding(self,X,px,py):
        inputX =np.array(X)
        for i in range(px):
            inputX=np.insert(inputX,0,0,axis=0)
            inputX=np.insert(inputX,inputX.shape[0],0,axis=0)
        for i in range(py):
            inputX=np.insert(inputX,0,0,axis=1)
            inputX=np.insert(inputX,inputX.shape[1],0,axis=1)
        return inputX
    
    def ConvolutionalLayer(self,X,filter_size=(5,5,3),stride_length=(1,1),no_filters=3,padding=0):
        inputX = np.array(X)
        print(X.shape[0],X.shape[1],X.shape[2])
        
        #padding = (filter_size[0]-1)/2
        
        W2 = ((X.shape[0] - filter_size[0] + 2*padding)/stride_length[0]) + 1
        H2 = ((X.shape[1] - filter_size[1] + 2*padding)/stride_length[1]) + 1
        output=np.zeros(shape=(W2,H2,no_filters))
        print (W2,H2,no_filters)
        #Add Padding to original image
        inputX=self.padding(inputX,padding,padding)

        #Always add padding to ensure that the output size is same as input size spatially .
        #Generally this gives better results as compared to no padding
        kernellist = list()
        biaseslist = list()
        for filters in range(no_filters):
            x,x1=0,0
            #mu,sigma=0,0.1
            #kernel = np.random.normal(mu,sigma,filter_size)
            mean =np.array([0 for i in range(int(filter_size[2]))])
            z = np.array([1 for i in range(filter_size[2])])
            cov = np.diag(z)
            kernel = np.random.multivariate_normal(mean,cov,(filter_size[0],filter_size[1]))
            kernellist.append(kernel)
            #print kernel.shape
            bias = np.random.randn()
            biaseslist.append(bias)
            while((x+filter_size[0])<=inputX.shape[0]):
                y,y1=0,0
                while((y+filter_size[1])<=inputX.shape[1]):
                    value = bias
                    for l in range(filter_size[0]):
                        for k in range(filter_size[1]):
                            for m in range(filter_size[2]):
                                value=value+kernel[l][k][m]*inputX[x+l][y+k][m]
                    output[x1][y1][filters]=self.sigma(value)
                    y+=stride_length[1]
                    y1+=1
                x+=stride_length[0]
                x1+=1
        return (output,kernellist,biaseslist)


    def PoolingLayer(self,X,pool_size=(2,2),pool_type="max"):
        #Can add a stride here as well 
        inputX=np.array(X)
        #W2 = (W1 - F)/S +1
        #H2 = (H1 - F)/S +1
        #D2 = D1
        #Not common to use zeropadding for pooling layers.
        # Generally F=2,S=2
        # or        F=3,S=2
        
        x1=math.ceil(1.0*inputX.shape[0]/pool_size[0])
        y1=math.ceil(1.0*inputX.shape[1]/pool_size[1])
        padX = (pool_size[0]*x1)-inputX.shape[0]
        padY = (pool_size[1]*y1)-inputX.shape[1]
        #for i in range(inputX.shape[2]):
         #   for j in range(padX):
          #      for k in range(padY):
           #         np.append(
        
        # Add Padding
        for i in range(int(padX)):
            inputX=np.insert(inputX,inputX.shape[0],0,axis=0)
        for i in range(int(padY)):
            inputX=np.insert(inputX,inputX.shape[1],0,axis=1)

        
        output = np.zeros((x1,y1,inputX.shape[2]))
        for layer in range(inputX.shape[2]):
            for x in range(int(x1)):
                for y in range(int(y1)):
                    try:
                        l=[inputX[pool_size[0]*x + i][pool_size[1]*y + j][layer] for i in range(pool_size[0]) for j in range(pool_size[1])]
                        if(pool_type == "max"):
                            output[x][y][layer] = max(l)
                        elif(pool_type == "l2"):
                            l2=[i**2 for i in l]
                            output[x][y][layer]=math.sqrt(sum(l2))
                    except:
                        continue
        return output
                            
    def FullyConnectedLayer(self,X,no_classes,Weights,biases):
        predictedclass = 1
        outputY = np.zeros((no_classes))
        #if X is 3D then size of weights will be 4D with size being X.shape x no_classes since every neuron in X is connected to every neuron in the next layer
        maxval= 0
        for i in range(no_classes):
            output = biases[i]
            for x in range(Weights.shape[0]):
                for y in range(Weights.shape[1]):
                    for z in range(Weights.shape[2]):
                        output+=Weights[x][y][z][i]*X[x][y][z]
            outputY[i]=self.sigma(output)
            if(outputY[i] > maxval):
                maxval = outputY[i]
                predictedclass= i+1
        return (predictedclass,outputY)
    

def batch_normalize(X):
    inputX = np.array(X)
    output = np.zeros((inputX.shape))
    for i in range(inputX.shape[2]):
        mean =0.0
        for x in range(inputX.shape[0]):
            for y in range(inputX.shape[1]):
                mean+=inputX[x][y][i]
        mean = 1.0*mean/(inputX.shape[0]*inputX.shape[1])
        variance= 0.0
        for x in range(inputX.shape[0]):
            for y in range(inputX.shape[1]):
                variance += (inputX[x][y][i] - mean)**2
        sd = 1.0*variance/(inputX.shape[0]*inputX.shape[1])
        sd = math.sqrt(sd)
        for x in range(inputX.shape[0]):
            for y in range(inputX.shape[1]):
                output[x][y][i] = 1.0*(inputX[x][y][i]-mean)/sd
    return output

def rotate180(kernel):
    kernelnew = np.zeros(kernel.shape)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                kernelnew[i][j][k] = kernel[kernel.shape[0]-1-i][kernel.shape[1]-1-j][k]
    return kernelnew

def convolve(X,Y):
    size1 = ((X.shape[0] - Y.shape[0])/stride_length[0]) +1
    size2 = ((X.shape[1] - Y.shape[1])/stride_length[1]) +1
    convolved = np.zeros((size1,size2))
    for i in range(size1):
        for j in range(size2):
            for k in range(Y.shape[0]):
                for l in range(Y.shape[1]):
                    convolved[i][j]+=X[i+k][j+l] * Y[k][l]
    return convolved

def BackPropagation(network,yactual,ypredicted,z,Weights,biases,kernellist,biaseslist,poolLayer,convLayer):
    #Error in Last FC Layer
    delta4 = ypredicted - yactual
    #Error in layer produced after pooling
    delta3 = np.zeros(poolLayer.shape)
    for i in range(delta3.shape[0]):
        for j in range(delta3.shape[1]):
            for k in range(delta3.shape[2]):
                sum=0
                for x in range(len(ypredicted)):
                    sum+=Weights[i][j][k][x]*delta4[x]*poolLayer[i][j][k]*(1-poolLayer[i][j][k])
                delta3[i][j][k] =sum
    delta_weights = np.zeros(Weights.shape)
    for i in range(delta3.shape[0]):
        for j in range(delta3.shape[1]):
            for k in range(delta3.shape[2]):
                for l in range(len(ypredicted)):
                    delta_weights[i][j][k][l] = delta4[l]*poolLayer[i][j][k]
    
    #delta2 will be found by extending this delta3 back to the pool size
    #delta2 is of size of the final convolution Layer //assuming stride=(1,1) delta2 = (W1-F,H1-F,no_filters)
    # delta2 is found as : if yijk(l) = max over p,q x_{i,j+p,k+q}, then the gradient for x_{ijk} is given as :
    # delta x_{ijk}(l) =0 except for delta x_{i,j+p',k+q'}(l) = delta y_{ijk}(l) where p',q' = argmax x_{i,j+p,k+q}
    delta2 = np.zeros((convLayer.shape[0],convLayer.shape[1],convLayer.shape[2]))
    if(pool_type == "max"):
        for i in range(poolLayer.shape[0]):
            for j in range(poolLayer.shape[1]):
                for k in range(poolLayer.shape[2]):
                    for x in range(pool_size[0]):
                        for y in range(pool_size[1]):
                            if (poolLayer[i][j][k] == convLayer[(pool_size[0]*i)+x][(pool_size[1]*j)+y][k]):
                                delta2[(pool_size[0]*i)+x][(pool_size[1]*j)+y][k] = delta3[i][j][k]
                            else:
                                delta2[(pool_size[0]*i)+x][(pool_size[1]*j)+y][k] = 0
    
    #d is the list of delta1 values for each filter/kernel used 
    d = list()
    # delta2 for full convolution
    
    delta2new = network.padding(delta2,filter_size[0]-1,filter_size[1]-1)

    #list of kernels and biases for convolution layer: kernellist and biaseslist

    for r in range(delta2new.shape[2]):
        kernel = kernellist[r]
        #Filter for full convolution
        kernelnew = rotate180(kernel)
        delta1= np.zeros(z.shape)
        for k in range(delta1.shape[2]):
            for i in range(delta1.shape[0]):
                for j in range(delta1.shape[1]):
                    sum=biaseslist[r]
                    for x in range(kernelnew.shape[0]):
                        for y in range(kernelnew.shape[1]):
                            sum+=delta2new[i+x][j+y][k]*kernelnew[x][y][k]
                    delta1[i][j][k] = sum
        d.append(delta1)
    kernels = np.zeros((no_filters,filter_size[0],filter_size[1],z.shape[2]))
    delta1s = np.zeros((no_filters,z.shape[0],z.shape[1],z.shape[2]))

    #delta of the loss function w.r.t ith input channel(here 3 input channels RGB)
    deltax_i = np.zeros(d[0].shape)
    for i in range(no_filters):
        deltax_i = deltax_i + d[i]
        
    for i in range(no_filters):
        kernels[i] = kernellist[i]
        delta1s[i] = d[i]
    deltaW = np.zeros((no_filters,filter_size[0],filter_size[1],z.shape[2]))
    deltabias= np.zeros((no_filters))
    for i in range(no_filters):
        for j in range(deltaW.shape[1]):
            for k in range(deltaW.shape[2]):
                for l in range(deltaW.shape[3]):
                    deltaW[i][j][k][l] = 0
                    deltabias[i]=0
                    for x in range(delta2.shape[0]):
                        for y in range(delta2.shape[1]):
                            deltaW[i][j][k][l]+=delta2[x][y][i]*z[x+(stride_length[0]*j)][y+(stride_length[1]*k)][l]
                            deltabias[i] +=delta2[x][y][i]
    #delWj(l) = sum over i delyj(l) * xi~ where xi~ is the row/column flipped version of xi and Wj is the jth kernel
    # deltaW_j = np.zeros(deltaW.shape)
    #x_flipped = rotate180(z)
    #for i in range(deltaW_j.shape[0]):
    #   for j in range(deltaW_j.shape[3]):
    #        deltaW_j[i,:,:,j] = convolve(delta2new[:,:,i],x_flipped[:,:,j])
    return (delta_weights,deltaW,deltabias)

def gradient_descent(Weights,Biases,delta_weights,kernellist,biaseslist,deltaW,deltabias,learning_rate):
    kernellist = kernellist - (learning_rate*deltaW)
    biaseslist = biaseslist - (learning_rate*deltabias)
    Weights = Weights - (learning_rate*delta_weights)
    #Biases = Biases - (learning_rate*delta_biases)
    return (kernellist,biaseslist,Weights,Biases)    
    
im = cv2.imread("bucky.jpg")
#cv.imshow('frame1',im)
#cv.waitKey(0)
#cv.destroyAllWindows()
z = np.array(im)
z = batch_normalize(z)
network = ConvolutionalNN()
#padding =( filter_size - 1 )/2
(convLayer,kernellist,biaseslist) = network.ConvolutionalLayer(z,filter_size,stride_length,no_filters,padding)
poolLayer = network.PoolingLayer(convLayer,pool_size,pool_type)
print (poolLayer)
#For final Fully Connected Layer, X is the layer just before it.
mu =np.array([0 for i in range(no_classes)])
cov2 = np.array([1 for i in range(no_classes)])
cov = np.diag(cov2)
Weights = np.random.multivariate_normal(mu,cov,poolLayer.shape)
biases = np.random.multivariate_normal(mu,cov)                    
FCLayer = network.FullyConnectedLayer(poolLayer,no_classes,Weights,biases)
predicted = FCLayer[1]
predicted.resize(no_classes)
Y =np.zeros(no_classes)
#for i in range(no_epochs):
# for j in range(no_train_examples):
# do Back Prop for each training example and then apply gradient descent for each training example i.e Stochastic Gradient descent
(delta_weights,deltaW,deltabias) = BackPropagation(network,Y,predicted,z,Weights,biases,kernellist,biaseslist,poolLayer,convLayer)
print (delta_weights.shape,Weights.shape)
print (deltaW.shape)
print (deltabias.shape)
kernels = np.zeros((len(kernellist),kernellist[0].shape[0],kernellist[0].shape[1],kernellist[0].shape[2]))
Biases = np.zeros((len(biaseslist)))
for i in range(len(kernellist)):
    Biases[i] = biaseslist[i]
    kernels[i] = kernellist[i]
(kernels,Biases,Weights,biases)= gradient_descent(Weights,biases,delta_weights,kernels,Biases,deltaW,deltabias,learning_rate)

