# Import the image and convert to RGB
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

coding_path = os.path.abspath("blurring.py")
image_path = os.path.join(os.path.dirname(coding_path), 'images/text.jpg')
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the image with different kernel sizes
kernels = [5, 11, 17]

fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 10))
fig.suptitle('Average Blurring with Different Kernel Sizes')
axs[0].imshow(img)
axs[0].axis('off')
axs[0].set_title('Original Image') # Shows the original images at the first
for s in range(3):
    kernelSize = kernels[s]
    img_blurred = cv2.blur(img, ksize = (kernelSize, kernelSize))
    ax = axs[s+1]
    ax.imshow(img_blurred)
    ax.axis('off')
    ax.set_title('(%d, %d) kernel' % (kernelSize, kernelSize))
plt.show()

# Different method of Blurring
img_0 = cv2.blur(img, ksize = (7, 7))
img_1 = cv2.GaussianBlur(img, ksize = (7, 7), sigmaX = 0)
img_2 = cv2.medianBlur(img, 7)
img_3 = cv2.bilateralFilter(img, 7, sigmaSpace = 75, sigmaColor = 75)

# Plot the images
images = [img_0, img_1, img_2, img_3]
fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 10))
fig.suptitle('Different Methods of Blurring')
for ind, p in enumerate(images):
    ax = axs[ind]
    ax.imshow(p)
    ax.axis('off')
blurs = ['Average', 'Gaussian', 'Median', 'Bilateral']
for ind, b in enumerate(blurs):
    ax = axs[ind]
    ax.set_title(b + ' Blur')
plt.show()
