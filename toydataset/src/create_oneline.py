'''
generate toy datasets with one line on Gaussian noise background


Save annotation in .csv file
save file name in train.txt
image_name
image_name	height	width	left_x	left_y	right_x	right_y
0006/1005869373_b4ced845a2_o.jpg	1800	2400	-1200	32.590726	1200	-135.995721
Note the origin is at the center!
'''
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import gc

from src.functionals import normal2leftright
from src.utils import make_directory

np.random.seed()

OOD_FLAG = False # set True for OOD test set !!!!!!!!

# create a white image
img = np.ones((224, 224, 3), dtype = np.uint8)
img = 255* img

num_img = 500

Nx = 224 # width
Ny = 224 # height
# Line thickness (larger value gives a wider line)
thickness = 1.0
# Noise scaling
noise_scale = 0.1

# annotation
left_x = -Nx / 2
right_x = Nx / 2

# train/val/test split
train_ratio = 0.7
val_ratio = 0.1

noise_img_path = 'toydataset/oneline/images'
white_img_path = 'toydataset/white'
make_directory(noise_img_path)
make_directory(white_img_path)
make_directory('toydataset/oneline/images')
make_directory('toydataset/oneline/split')

with open('toydataset/oneline/toydata.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'height', 'width', 'left_x',	'left_y', 'right_x', 'right_y'])

xc,yc = np.meshgrid(np.arange(0,Nx,1)-Nx/2,np.arange(0,Ny,1)-Ny/2)


for i in range(num_img):
    # path
    img_name = f'{i}.jpg'
    # Generate coordinate grid
    
    rho = (np.random.rand()-0.5)*100
    if OOD_FLAG:
        theta = (np.random.rand() -0.5) * np.pi/6 #[-Pi/6, Pi/6)  
    else:    
        # for OoDtest test
        if i < num_img/2:                         #  [-pi/2, -pi/6) and (pi/6, pi/2]
            theta = np.random.rand()*(1/2-1/6)*np.pi + np.pi/6 
        else:
            theta = np.random.rand()*(-1/2+1/6)*np.pi - np.pi/6
    
    l = np.array([-np.sin(theta),np.cos(theta),-rho])  

    # Signed distance to line;the minimum distance gets higher value
    sd = l[0]*xc+l[1]*yc+l[2]

    img = np.cos(np.minimum(np.abs(sd)/thickness,np.pi/2.0))**2 # cosine is for the line to fade smoothly

    # Insert noise
    img_noise = np.maximum(img,noise_scale*np.random.rand(Nx,Ny))

    # write annotation
    left_y, right_y = normal2leftright(slope=theta, offset=rho/Nx) # ambiguity???
    with open('toydataset/oneline/toydata.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([img_name, Ny, Nx, left_x, left_y, right_x, right_y])  

    # save Gaussian background
    plt.figure(figsize=(4, 4))
    plt.imshow(img_noise, cmap='binary')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(noise_img_path, img_name), dpi=56, pad_inches = 0)
    plt.close()
    del img_noise
    gc.collect()        
    
    # save white background
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='binary')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(white_img_path, img_name), dpi=56, pad_inches = 0)
    plt.close()
    del img
    gc.collect() 
    
    # write file path  
    if i <= train_ratio * num_img:
        with open('toydataset/oneline/split/train.txt', 'a') as f:
            f.write(f'{img_name}\n')
    elif train_ratio*num_img <= i <= (val_ratio+train_ratio)*num_img:
        with open('toydataset/oneline/split/val.txt', 'a') as f:
            f.write(f'{img_name}\n')
    else:
        with open('toydataset/oneline/split/test.txt', 'a') as f:
            f.write(f'oneline/{img_name}\n')

