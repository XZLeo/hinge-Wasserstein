'''
For the training set, mixture of 1 line and 2 line
one line per image and two lines per image
two random lines, the second is the annotation
Save annotation in .csv file
save file name in train.txt
image_name
image_name	height	width	left_x	left_y	right_x	right_y
0006/1005869373_b4ced845a2_o.jpg	1800	2400	-1200	32.590726	1200	-135.995721
Note the origin is at the center!

Jiter the label on the parameter space
'''
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import gc

from src.functionals import normal2leftright
from src.utils import make_directory

np.random.seed()

# create a white image
img = np.ones((224, 224, 3), dtype = np.uint8)
img = 255* img

num_lines = 2
num_img = (5000, 2500)

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

THETA_NOISE = True #!!!!!!!!!change the img path accordingly!!!!!!!
RHO_NOISE = True #!!!!!!!!!change the img path accordingly!!!!!!!

img_path = 'toydataset/theta_rho_noise/'
make_directory(img_path)
make_directory(f'{img_path}/images')
make_directory(f'{img_path}/split')

with open(f'{img_path}/twolines.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'height', 'width', 'left_x',	
                     'left_y_1', 'left_y_2', 'right_x', 'right_y_1', 'right_y_2'])

xc,yc = np.meshgrid(np.arange(0,Nx,1)-Nx/2,np.arange(0,Ny,1)-Ny/2)

for line in range(num_lines):
    count = 0
    for i in range(num_img[line]):
        # path
        img_name = f'{line+1}lines_{i}.png'
        # Generate coordinate grid
        
        img = np.zeros((Nx, Ny))
        
        line_dict = {'left_y':[], 'right_y':[]}
        for num in range(line+1):   # loop for each image 
            # Generate a random line l
            rho = (np.random.rand()-0.5)*100 # pixel  [-50, 50]
            theta = (np.random.rand() -0.5)*np.pi #[-Pi/2, Pi/2)  # change to Gaussian
            
            if THETA_NOISE:
                theta_jitter = theta + np.random.normal(loc=0, scale=0.1) #0.16034rad= 6 deg
            else:
                theta_jitter = theta
            if RHO_NOISE: 
                rho_jitter = rho + np.random.normal(loc=0, scale=0.5*Nx) #0.9474
            else:
                rho_jitter = rho
            left_y, right_y = normal2leftright(slope=theta_jitter, offset=rho_jitter/Nx) 
            line_dict['left_y'].append(left_y)
            line_dict['right_y'].append(right_y)            
                    
            l = np.array([-np.sin(theta),np.cos(theta),-rho])  

            # Signed distance to line;the minimum distance gets higher value
            sd = l[0]*xc+l[1]*yc+l[2]

            new_line = np.cos(np.minimum(np.abs(sd)/thickness,np.pi/2.0))**2 # cosine is for the line to fade smoothly
            img = np.maximum(img, new_line)
        
        if line == 0:
            line_dict['left_y'].append(float('inf'))
            line_dict['right_y'].append(float('inf'))
        
        # Insert noise
        img = np.maximum(img,noise_scale*np.random.rand(Nx,Ny))

        # write annotation
        with open(f'{img_path}/twolines.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([img_name, Ny, Nx, left_x, line_dict['left_y'][0], line_dict['left_y'][1],
                             right_x, line_dict['right_y'][0], line_dict['right_y'][1]])
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='binary')
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(f'{img_path}/images', img_name), dpi=56, pad_inches = 0)
        plt.close()
        del img
        gc.collect()        
      
        # write file path  
        count += 1
        if count <= train_ratio * num_img[line]:
            with open(f'{img_path}/split/train.txt', 'a') as f:
                f.write(f'{img_name}\n')
        elif train_ratio*num_img[line] <= count <= (val_ratio+train_ratio)*num_img[line]:
            with open(f'{img_path}/split/val.txt', 'a') as f:
                f.write(f'{img_name}\n')
        else:
            with open(f'{img_path}/split/test.txt', 'a') as f:
                f.write(f'{img_name}\n')
