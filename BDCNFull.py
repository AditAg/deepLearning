# here noise is added to original image patch and not binarized image patch else it gives an np array of all 1's
#after adding this noise the noisy image is trained with target as clean binary image
from __future__ import absolute_import
from keras import backend as K
#from keras.layers import Input, Dense,Convolution2D, MaxPooling2D,Dropout, Flatten, Merge, Reshape, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense,Flatten,Activation,Dropout,Reshape
#cant use conv1d here since in that filter_size is 1d and cannot be 2D
from keras.models import Model,Sequential    
# SGD can be used with momentum and nesterov updates, RMSprop , Adagrad , Adadelta, Adam, Adamax, Nadam TFOptimizer are also provided
from keras import optimizers
from keras.optimizers import SGD
import numpy as np
import cv2
import os
import random
import tensorflow as tf
import math
import gc
batch_size=30
no_epochs = 1
l=0.5                                                          # lambda parameter for loss function
def indicator_function(x):
	return 1
def custom_objective_single(y1,y2):
	return (-1.0*(1+l*indicator_function(y1))*(K.dot(K.transpose(K.reshape(y1,(3136,1))),K.log(K.reshape(y2,(3136,1))))))
	
def custom_objective(y_true, y_pred):
    	return K.sum(tf.map_fn(lambda x: custom_objective_single(x[0], x[1]), (y_true, y_pred), dtype=tf.float32))
	#loss = tf.diag_part(tf.matmul(y_true,tf.transpose(tf.log(y_pred))))
	#l2 = tf.scalar_mul((-1.0*(1+l*indicator_function(y_true))),loss)
	#return K.mean(l2)	
	#return K.mean(-1.0*(1+l*indicator_function(y_true))*(K.dot(K.transpose(y_true),K.log(y_pred))))


def get_data():
	writers = {}
	file1 = open('IAM Dataset/ascii/forms.txt','r')
	z = file1.read()
	file1.close()
	data = z.strip().split('\n')
	writers={}
	length = 0
	for i in range(len(data)):
		if(data[i][0]=='#' or data[i][0] >'d'):
			continue
		else:
			d = data[i].strip().split(' ')
			length+=1
			if(d[1] not in writers.keys()):
				writers[d[1]]=[]
			writers[d[1]].append(d[0])

	docs = np.zeros((length,424,424))
	y_labels = np.zeros((length))
	k=0
	for i in writers.keys():
		for j in range(len(writers[i])):
			string = 'IAM Dataset/forms/'+str(writers[i][j]) + '.png'
			img =cv2.imread(string,0)
			img = img[650:3100,300:2450]
			img = cv2.resize(img,(424,424))
			docs[k]=img
			y_labels[k] = int(i)
			k = k+1
	return (docs,y_labels)

def generate_patches(train_data,y_labels):
	patches = list()
	binarized_patches=list()
	y_vals = list()
	for k in range(len(train_data)):
		im = train_data[k]
		yval = y_labels[k]
		#Add padding to image so that the entire image can be obtained as patches
		z1 = (im.shape[0]-56)%46
		if(z1 == 0):
			px = 0
		else:
			px = ((int((im.shape[0]-56)/46)+1)*46) + 56 - im.shape[0]
		if((im.shape[1]-56)%46 == 0):
			py = 0
		else:
			py = ((int((im.shape[1]-56)/46)+1)*46) + 56 - im.shape[1]
		for i in range(int(px/2)):
    			im = np.insert(im,0,0,axis=0)
    			im = np.insert(im,im.shape[0],0,axis=0)
		if(px%2!=0):
			im = np.insert(im,0,0,axis=0)
		for i in range(int(py/2)):
			im = np.insert(im,0,0,axis=1)
			im = np.insert(im,im.shape[1],0,axis=1)
		if(py%2!=0):
			im = np.insert(im,0,0,axis=1)
		#take image patches of size 56X56
		y=0
		z = np.amax(im)
		z = 0.95*z                                # 0.45 of the document maximum(can change it since it is giving 1 for all pixels)
		image = np.zeros((56,56))
		image1= np.zeros((56,56))
		while(y + 56 < im.shape[1]):
			x = 0
			while(x + 56 < im.shape[0]):
				image = im[x:x+56,y:y+56]
				#binarize_image_patch       # can do it later as well by returning the document maximums as well
				np.copyto(image1,im[x:x+56,y:y+56])    # cannot do image1=image else below it changes both image and image1				
				image1[image1<z] = 0				
				image1[image1>=z] = 1         # create a new np array if original patches are also required
				patches.append(image)
				binarized_patches.append(image1)
				y_vals.append(yval)
				x+=46       # to overlap by 10    (overlap of 20 for patches of size 120X120)
			y+=46
	image_patches = np.zeros((len(patches),56,56))
	binarized_image_patches=np.zeros((len(patches),56,56))
	y_values = np.zeros((len(patches)))	
	for i in range(len(patches)):
		image_patches[i]=patches[i]
		binarized_image_patches[i]=binarized_patches[i]
		y_values[i] = y_vals[i]
	return (image_patches,y_values,binarized_image_patches)
	
def generate_image_paths(path_to_directory):
    f=[]
    for (dirpath,dirnames,filenames) in os.walk(path_to_directory):
        for filename in filenames:
            filepath = os.path.join(dirpath,filename)
            f.append(filepath)
    return f

def add_noise(image,images_list):
    n = len(images_list)
    no = random.randint(0,n-1)
    noise_image = cv2.imread(images_list[no],0)
    #changing intensity
    noise_image = noise_image + random.randint(-255,255)
    noise_image = noise_image%256
    # could also change intensity of every pixel individually
    #changing contrast
    noise_image =np.rint(noise_image * random.uniform(0,5))
    noise_image =(noise_image%256)
    #add other noises to the noise image
    
    noise_image = cv2.resize(noise_image,(56,56))
    x = np.ones(noise_image.shape)
    x*=255
    x1 = (2.0*255) - image - noise_image
    noisy_image = np.minimum(x1,x)
    noisy_image = (1.0/255)*noisy_image                      # since it is 1/255 so on a binary image 2*255 -(0 or 1)-(something b/w 0 and 255) is always >255 and so min(x1,x) always gives 255 so that noisy_image always gives 1 for every pixel

    return noisy_image

(train_data,y_values) = get_data()
(image_patches,y_vals,binarized_image_patches) = generate_patches(train_data,y_values)
#add noise to the binary patches
noisy_images = np.zeros(image_patches.shape)
random_image_paths = generate_image_paths("images")   # random images for noise 
for i in range(image_patches.shape[0]):
    	#x = image_patches[i].flatten()
    	#for j in range(x.shape[0]):
        # 	if(x[j]>=z):
        #    		x[j]=1
        #	else:
        #    		x[j]=0
    	#binarized_image = x.reshape((image_patches.shape[1],image_patches.shape[2],image_patches.shape[3]))
     
    	#Add noise to the binarized image
    	noisy_image = add_noise(image_patches[i],random_image_paths)                      
	#noise shouldn't be applied to binarized image patch else it gives a noisy image of all 1's
    	noisy_images[i] = noisy_image

# Apply the convolutional Neural Network to the noisy images
# Train the denoiser on the input patches
noisy_images = np.reshape(noisy_images,(noisy_images.shape[0],noisy_images.shape[1],noisy_images.shape[2],1))
# for the denoiser the y values are clean binary image patches
y_true = binarized_image_patches.flatten()
y_true = np.reshape(y_true,(binarized_image_patches.shape[0],binarized_image_patches.shape[1]*binarized_image_patches.shape[2]))
print(noisy_images)
print(y_true)
model = Sequential()
model.add(Conv2D(128,(6,6),strides=(1,1),use_bias=True,input_shape=(56,56,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
model.add(Conv2D(128,(4,4),strides=(1,1),use_bias=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
model.add(Conv2D(128,(4,4),strides=(1,1),use_bias=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128,(4,4),strides=(1,1),use_bias=True))
model.add(Activation('relu'))
model.add(Flatten())                                                #Always remember to add flatten before the 1st dense layer else it gives error#
model.add(Dense(1024,activation='relu'))
model.add(Dense(2048,activation='relu'))
model.add(Dense(3136,activation='sigmoid'))
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss=custom_objective,optimizer=sgd,metrics=['accuracy'])
model.fit(noisy_images,y_true,batch_size=batch_size,epochs=no_epochs,shuffle=True)

#When we test above on test data: the output will be for every patch of the image
image_test = cv2.imread('IAM Dataset/forms/d07-102.png',0)    # here we should instead have a loop to get our test data
image_test = image_test[650:3100,300:2450]
img = cv2.resize(image_test,(424,424))
new_image= np.zeros((504,504))
for i in range(9):
	for j in range(9):
		start_x = i*46
		start_y = j*46
		image_patch = img[start_x:start_x+56,start_y:start_y+56]
		image_patch = np.resize(image_patch,(1,56,56,1))
		output = model.predict(image_patch,batch_size=1)                   #Numpy array of predictions
		print(output.shape)		
		new_image[i*56:56*(i+1),56*j:56*(j+1)] = output.reshape((56,56))
new_image = cv2.resize(new_image,(424,424))
print (new_image)
#Automated Segmentation ----- Corner Detection using bending value   See this paper   and then do segment generation and Pseudo Character segmentation

#Binarize the image and find the connected components
z = np.amax(new_image)
z = 0.9*z
binarized_image=np.zeros((new_image.shape))
binarized_image[new_image<z]=0
binarized_image[new_image>=z]=1
print (binarized_image)
#two-pass algorithm
linked={}
nextlabel=1
labels = np.zeros((binarized_image.shape))
parent={}
def find(X):
	if(parent[X]==X):
		return X
	else:
		return find(parent[X])

def union(X,Y):
	labels=set()
	for i in X:
		for j in range(len(Y)):
			xroot=find(i)
			yroot=find(Y[i])
			parent[xroot]=yroot
			labels.add(yroot)
	return labels 
		
for i in range(binarized_image.shape[0]):
	for j in range(binarized_image.shape[1]):
		if(binarized_image[i][j]!=0):
			#determine neighbours
			if(i==0 and j==0):
				neighbours=[]
			elif(i==0):
				neighbours=[(i,j-1)]
			elif(j==0):
				neighbours=[(i-1,j),(i-1,j+1)]
			elif(j==binarized_image.shape[1]-1):
				neighbours=[(i-1,j-1),(i-1,j),(i,j-1)]
			else:
				neighbours = [(i-1,j),(i-1,j+1),(i-1,j-1),(i,j-1)]
			actual_neighbours=[]
			for k in range(len(neighbours)):
				if(binarized_image[neighbours[k]]==binarized_image[i][j]):
					actual_neighbours.append(neighbours[k])
			if(len(actual_neighbours)==0):
				linked[nextlabel]={nextlabel}
				parent[nextlabel]=nextlabel
				labels[i][j]=nextlabel
				nextlabel+=1
			else:
				L=[]
				for k in range(len(actual_neighbours)):
					L.append(labels[actual_neighbours[k][0]][actual_neighbours[k][1]])
				labels[i][j] = min(L)
				for label in L:
					linked[label]=union(linked[label],L)
for i in range(binarized_image.shape[0]):
	for j in range(binarized_image.shape[1]):
		if(binarized_image[i][j]!=0):
			labels[i][j]=find(labels[i][j])

print (labels)												
#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)        #this only works for images i.e. where pixel values are between 0 and 255 it wont work for new_image which is probably in range -1 and 1
#Threshold on size(no. of pixels) and aspect ratio(w:h)

# Data Augmentation : adding rotation [-15,+15]; some skew (rho<1.5),
#intensity of foreground(pixel X i where i~N(1,0.1)) and scale [0.8,1.2])

#Then do further segmentation using seam, vert and TAS methods  csn ignore
#Then use Fiel's network(caffenet) for writer identification on the individual local patches
#obtained after segmentation and preprocessing above.
        
