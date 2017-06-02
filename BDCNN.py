import numpy as np
import cv2
import os
import random

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

    x = np.ones(noise_image.shape)
    x*=255
    x1 = (2*255) - image - noise_image
    noisy_image = np.minimum(x1,x)
    noisy_image = (1/255)*noisy_image
    return noisy_image

im = cv2.imread('bucky.jpg')     # Here open the image for the document instead of bucky.jpg
#Add padding to image so that the entire image can be obtained as patches
px = (im.shape[0]-56)%46
py = (im.shape[1]-56)%46
for i in range(int(px/2)):
    im = np.insert(im,0,0,axis=0)
    im = np.insert(im,im.shape[0],0,axis=0)
for i in range(int(py/2)):
    im = np.insert(im,0,0,axis=1)
    im = np.insert(im,im.shape[1],0,axis=1)

#take image patches of size 56X56
patches=[]
y=0
image = np.zeros((56,56,im.shape[2]))
while(y+56 < im.shape[1]):
    x = 0
    while(x+56 < im.shape[0]):
        image = im[x:x+56,y:y+56,:]
        patches.append(image)
        x+=46       # to overlap by 10    (overlap of 20 for patches of size 120X120)
    y+=46
image_patches = np.zeros((len(patches),56,56,im.shape[2]))
for i in range(len(patches)):
    image_patches[i]=patches[i]

#binarize image patches and add noise to them
noisy_images = np.zeros(image_patches.shape)
z = np.amax(im)
z = 0.45*z            #0.45 of the document maximum
random_image_paths = generate_image_paths(".\images")   # random images for noise 
for i in range(image_patches.shape[0]):
    x = image_patches[i].flatten()
    for j in range(x.shape[0]):
        if(x[j]>=z):
            x[j]=1
        else:
            x[j]=0
    binarized_image = x.reshape((image_patches.shape[1],image_patches.shape[2],image_patches.shape[3]))
    #Add noise to the binarized image
    noisy_image = add_noise(binarized_image,random_image_paths)
    noisy_images[i] = noisy_image

# Apply the convolutional Neural Network to the noisy images
# Train the denoiser on the input patches

#Automated Segmentation
#Threshold on size(no. of pixels) and aspect ratio(w:h)
# Data Augmentation : adding rotation [-15,+15]; some skew (rho<1.5),
#intensity of foreground(pixel X i where i~N(1,0.1)) and scale [0.8,1.2])

#Then do further segmentation using seam, vert and TAS methods  csn ignore
#Then use Fiel's network(caffenet) for writer identification on the individual local patches
#obtained after segmentation and preprocessing above.
        
