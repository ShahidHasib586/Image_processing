import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv.imread('1.jpg')
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