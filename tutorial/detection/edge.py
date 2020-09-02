import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.abspath('canny.py')
img_path = os.path.dirname(path) + '/images/giraffe.jpg'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Canny detection without blurring and just the median value for two thresholds
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))
axs[0].imshow(img)
axs[0].axis('off')
axs[0].set_title('Original Image')

axs[1].imshow(edges)
axs[1].axis('off')
axs[1].set_title('Canny without Blur')
plt.show()

# Set the lower and upper threshold
med_val = np.median(img)

lower = int(max(0, .7*med_val))
upper = int(min(255, 1.3*med_val))

# Blurring with ksize = 5
img_k5 = cv2.blur(img, ksize = (5, 5))

# Canny detection with different thresholds
edges_k5 = cv2.Canny(img_k5, threshold1 = lower, threshold2 = upper)
edges_k5_2 = cv2.Canny(img_k5, lower, upper+100)

# Blurring with kernel size = 9
img_k9 = cv2.blur(img, ksize = (9,9))

# Canny detection with different thresholds
edges_k9 = cv2.Canny(img_k9, lower, upper)
edges_k9_2 = cv2.Canny(img_k9, lower, upper+100)

# Plot the images
images = [edges_k5, edges_k5_2, edges_k9, edges_k9_2]
titles = ['ksize=(5,5)\nlower,upper', 'ksize=(5,5)\nlower,upper+100',
'ksize=(9,9)\nlower,upper', 'ksize=(9,9)\nlower,upper+100']
fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (20, 8))
for ind, p in enumerate(images):
    ax = axs[ind//2, ind%2]
    ax.imshow(p)
    ax.set_title(titles[ind], y =-0.2) # 'y' parameter is for placing titles below the images.
    ax.axis('off')
plt.show()
