# one possible modification is to do train model on 3D input
#another possible modification is to add a resize layer at last and change size to 56X56 and do sigmoid on greyscale of binary clean image 
#Other modifications are to change activation functions, dropout rate, optimizers (rsprop or adagrad etc)
from keras.layers import Input, Dense,Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Merge, Reshape, Activation
from keras.models import Model,Sequential    
# SGD can be used with momentum and nesterov updates, RMSprop , Adagrad , Adadelta, Adam, Adamax, Nadam TFOptimizer are also provided
from keras import optimizers
from keras.optimizers import SGD
import numpy as np
import cv2
import os
import random
import tensorflow as tf

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

	docs = np.zeros((length,424,424,3))
	y_labels = np.zeros((length))
	k=0
	for i in writers.keys():
		for j in range(len(writers[i])):
			string = 'IAM Dataset/forms/'+str(writers[i][j]) + '.png'
			img =cv2.imread(string)
			img = img[650:3100,300:2450,:]
			img = cv2.resize(img,(424,424))
			docs[k]=img
			y_labels[k] = int(i)
			k = k+1
	return (docs,y_labels)

def generate_patches(train_data,y_labels):
	patches = list()
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
		if((im.shape[0]-56)%46 == 0):
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
		z = 0.45*z                                  # 0.45 of the document maximum
		image = np.zeros((56,56,im.shape[2]))
		while(y + 56 < im.shape[1]):
			x = 0
			while(x + 56 < im.shape[0]):
				image = im[x:x+56,y:y+56,:]
				#binarize_image_patch       # can do it later as well by returning the document maximums as well
				image[image>=z] = 1         # create a new np array if original patches are also required
				image[image<z] = 0
				patches.append(image)
				y_vals.append(yval)
				x+=46       # to overlap by 10    (overlap of 20 for patches of size 120X120)
			y+=46
	image_patches = np.zeros((len(patches),56,56,train_data.shape[3]))
	y_values = np.zeros((len(patches)))	
	for i in range(len(patches)):
		image_patches[i]=patches[i]
		y_values[i] = y_vals[i]
	return (image_patches,y_values)
	
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
    noise_image = cv2.imread(images_list[no])
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
    x1 = (2*255) - image - noise_image
    noisy_image = np.minimum(x1,x)
    noisy_image = (1/255)*noisy_image
    return noisy_image

(train_data,y_values) = get_data()
(image_patches,y_vals) = generate_patches(train_data,y_values)
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
    	noisy_images[i] = noisy_image

# Apply the convolutional Neural Network to the noisy images
# Train the denoiser on the input patches
y_true = image_patches                                             # for the denoiser the y values are clean binary image patches
model = Sequential()
model.add(Convolution2D(128,(6,6),strides=(1,1),use_bias=True,input_shape=(56,56,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
model.add(Convolution2D(128,(4,4),strides=(1,1),use_bias=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
model.add(Convolution2D(128,(4,4),strides=(1,1),use_bias=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(128,(4,4),strides=(1,1),use_bias=True))
model.add(Activation('relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(2048,activation='relu'))
model.add(Activation('sigmoid'))


#Automated Segmentation
#Threshold on size(no. of pixels) and aspect ratio(w:h)
# Data Augmentation : adding rotation [-15,+15]; some skew (rho<1.5),
#intensity of foreground(pixel X i where i~N(1,0.1)) and scale [0.8,1.2])

#Then do further segmentation using seam, vert and TAS methods  csn ignore
#Then use Fiel's network(caffenet) for writer identification on the individual local patches
#obtained after segmentation and preprocessing above.
        
