import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.path.abspath('morphological.py')
dir_path = os.path.dirname(path)

img_path = dir_path + '/images/simpson.jpg'
img = cv2.imread(img_path)
img = ~img # inverting black and white

# Create erosion kernels (The erosion makes the object in white smaller)
kernel_0 = np.ones((9, 9), np.uint8) # Basic kernel
kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)) # Elipse kernel
kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)) # Cross kernel

kernels = [kernel_0, kernel_1, kernel_2]

# Plot the images
fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 8))
axs[0].imshow(img)
axs[0].axis('off')
for i in range(1, 4):
    ax = axs[i]
    img_copy = img.copy()
    img_copy = cv2.erode(img_copy, kernels[i-1], iterations = 3)
    ax.imshow(img_copy)
    ax.axis('off')
plt.show()


# Dilation (It makes the object in white bigger)
kernel = np.ones((9, 9), np.uint8)
img_dilate = cv2.dilate(img, kernel, iterations = 3)

plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray")
plt.subplot(1, 2, 2); plt.imshow(img_dilate, cmap="gray")
plt.show()


# Different types of morphological techniques
kernel = np.ones((9, 9), np.uint8)

img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# Plot the images
images = [img, img_open, img_close, img_gradient, img_tophat, img_blackhat]

fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 8))
for ind, p in enumerate(images):
    ax = axs[ind//3, ind%3]
    ax.imshow(p, cmap = 'gray')
    ax.axis('off')
plt.show()
