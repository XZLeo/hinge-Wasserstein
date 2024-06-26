'''
read white images generated by creat_oneline.py
add a second line on it
save two csv file:
one with the second line as the annotation
one with both lines as the annotation
'''
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import gc
import pandas as pd
from PIL import Image

from src.functionals import normal2leftright
from src.utils import make_directory

np.random.seed()


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

# one line dataset
oneline_path = 'toydataset/oneline/'
white_path = 'toydataset/white'
csv_path = os.path.join(oneline_path, 'toydata.csv')

twoline_path = 'toydataset/twolines'
make_directory(os.path.join(twoline_path, 'images'))
make_directory(os.path.join(twoline_path, 'split'))

with open('toydataset/twolines/oneline.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'height', 'width', 'left_x',	'left_y', 'right_x', 'right_y'])
    
with open('toydataset/twolines/twolines.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'height', 'width', 'left_x',	'left_y_1', 'right_x', 'right_y_1', 'left_y_2', 'right_y_2'])

xc,yc = np.meshgrid(np.arange(0,Nx,1)-Nx/2,np.arange(0,Ny,1)-Ny/2)

onelineFrame = pd.read_csv(csv_path)
num_img = onelineFrame.shape[0]

# read csv loop over white images
for row_idx, row in onelineFrame.iterrows():
    img_name, left_y_1, right_y_1 = row['image_name'], row['left_y'], row['right_y'] 
    # path
    white_img_path = os.path.join(white_path, img_name)
    # read white img
    white_img = (255 - np.array(Image.open(white_img_path).convert('L'), 'f'))/255
    white_img = white_img[::-1, :]

    # Generate coordinate grid
    rho = (np.random.rand()-0.5)*100
    theta = (np.random.rand() -0.5)*np.pi/2 #[-Pi/2, Pi/2)  # change to Gaussian
    l = np.array([-np.sin(theta),np.cos(theta),-rho])  

    # Signed distance to line;the minimum distance gets higher value
    sd = l[0]*xc+l[1]*yc+l[2]

    img = np.cos(np.minimum(np.abs(sd)/thickness,np.pi/2.0))**2 # cosine is for the line to fade smoothly
    img = np.maximum(img, white_img)
  
    # Insert noise
    img_noise = np.maximum(img,noise_scale*np.random.rand(Nx,Ny))

    # write annotation
    left_y_2, right_y_2 = normal2leftright(slope=theta, offset=rho/Nx) 
    with open('toydataset/twolines/oneline.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([img_name, Ny, Nx, left_x, left_y_2, right_x, right_y_2])
        
    with open('toydataset/twolines/twolines.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([img_name, Ny, Nx, left_x, left_y_1, right_x, right_y_1, left_y_2, right_y_2])

    plt.figure(figsize=(4, 4))
    plt.imshow(img_noise, cmap='binary')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(os.path.join(twoline_path, 'images'), img_name), dpi=56, pad_inches = 0)
    plt.close()
    del img_noise
    gc.collect()        
    
    # write file path  
    if row_idx <= train_ratio * num_img:
        with open('toydataset/twolines/split/train.txt', 'a') as f:
            f.write(f'{img_name}\n')
    elif train_ratio*num_img <= row_idx <= (val_ratio+train_ratio)*num_img:
        with open('toydataset/twolines/split/val.txt', 'a') as f:
            f.write(f'{img_name}\n')
    else:
        with open('toydataset/twolines/split/test.txt', 'a') as f:
            f.write(f'{img_name}\n')
 