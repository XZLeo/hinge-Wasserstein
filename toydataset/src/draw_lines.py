'''
example for anti aliasing drawing a second line on the white background
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = (255 - np.array(Image.open('toydataset/white/1.jpg').convert('L'), 'f'))/255
img = img[::-1, :]
plt.figure()
plt.imshow(img, cmap='binary')
plt.show()

# Image size
Nx = 224
Ny = 224
# Line thickness (larger value gives a wider line)
thickness = 1.0
# Noise scaling
noise_scale = 0.1

# Generate coordinate grid
xc,yc = np.meshgrid(np.arange(0,Nx,1)-Nx/2,np.arange(0,Ny,1)-Ny/2)

# Generate a random line l
rho = (np.random.rand()-0.5)*30
theta = (np.random.rand() -0.5)*np.pi #[-Pi/2, Pi/2)  # change to Gaussian
l = np.array([-np.sin(theta),np.cos(theta),-rho])

# Signed distance to line
sd = l[0]*xc+l[1]*yc+l[2]

new_img = np.cos(np.minimum(np.abs(sd)/thickness,np.pi/2.0))**2
img = np.maximum(img, new_img)

# Insert noise # don't add two images but use maximum operator!!
img = np.maximum(img,noise_scale*np.random.rand(Nx,Ny))

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap='binary')
plt.title('Random anti-aliased line in range [0,1]')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0, 0)
plt.savefig('toydataset/img/toy2.jpg', dpi=56, pad_inches = 0)
