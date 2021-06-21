# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:26:56 2021

@author: shahi
"""
#optimal conny
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('opnCV.jpg',0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Optimal Conny'), plt.xticks([]), plt.yticks([])
plt.show()

#roberts

#def filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None)
# Read image
img = cv.imread('opnCV.jpg', cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Roberts operator
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)

x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
y = cv.filter2D(grayImage, cv.CV_16S, kernely)
# Turn uint8, image fusion
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
# Display graphics
titles = ['The original image', 'Roberts operator']
images = [rgb_img, Roberts]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()



# Grayscale processing image
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#prewitt operator
# Read image
img = cv.imread('opnCV.jpg', cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Grayscale processing image
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Prewitt operator
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)

x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
y = cv.filter2D(grayImage, cv.CV_16S, kernely)

# Turn uint8, image fusion
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

# Used to display Chinese labels normally
plt.rcParams['font.sans-serif'] = ['SimHei']

# Display graphics
titles = ['The original image', 'Prewitt operator']
images = [rgb_img, Prewitt]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()




#Sobel operator



# Read image
img = cv.imread('opnCV.jpg', cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Grayscale processing image
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Sobel operator
x = cv.Sobel(grayImage, cv.CV_16S, 1, 0)
y = cv.Sobel(grayImage, cv.CV_16S, 0, 1)

# Turn uint8, image fusion
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

# Used to display Chinese labels normally
plt.rcParams['font.sans-serif'] = ['SimHei']

# Display graphics
titles = ['The original image', 'Sobel operator']
images = [rgb_img, Sobel]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
#2nd order
#Laplacian operator



# Read image
img = cv.imread('opnCV.jpg', cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Grayscale processing image
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacian
dst = cv.Laplacian(grayImage, cv.CV_16S, ksize = 3)
Laplacian = cv.convertScaleAbs(dst)

# Used to display Chinese labels normally
plt.rcParams['font.sans-serif'] = ['SimHei']

# Display graphics
titles = ['The original image', 'Laplacian operator']
images = [rgb_img, Laplacian]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

#laplacian of gaussian


# Read image
img = cv.imread('opnCV.jpg')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Grayscale processing image
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Gaussian filter
gaussian_blur = cv.GaussianBlur(gray_image, (3, 3), 0)

# Roberts operator
kernelx = np.array([[-1, 0], [0, 1]], dtype = int)
kernely = np.array([[0, -1], [1, 0]], dtype = int)
x = cv.filter2D(gaussian_blur, cv.CV_16S, kernelx)
y = cv.filter2D(gaussian_blur, cv.CV_16S, kernely)
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

# Prewitt operator
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv.filter2D(gaussian_blur, cv.CV_16S, kernelx)
y = cv.filter2D(gaussian_blur, cv.CV_16S, kernely)
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

# Sobel operator
x = cv.Sobel(gaussian_blur, cv.CV_16S, 1, 0)
y = cv.Sobel(gaussian_blur, cv.CV_16S, 0, 1)
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

# Laplace algorithm
dst = cv.Laplacian(gaussian_blur, cv.CV_16S, ksize = 3)
Laplacian = cv.convertScaleAbs(dst)

# Display image
titles = ['Source Image', 'Gaussian Image', 'Roberts Image',
          'Prewitt Image','Sobel Image', 'Laplacian Image']
images = [rgb_img, gaussian_blur, Roberts, Prewitt, Sobel, Laplacian]
for i in np.arange(6):
   plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
   plt.title(titles[i])
   plt.xticks([]), plt.yticks([])
plt.show()

'''
#difference of gaussian


img = cv.imread('opnCV.jpg')
 
# Apply 3x3 and 7x7 Gaussian blur
low_sigma = cv.GaussianBlur(img,(3,3),0)
high_sigma = cv.GaussianBlur(img,(5,5),0)
 
# Calculate the DoG by subtracting
dog = low_sigma - high_sigma

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
'''










































