import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.abspath('corner.py')
img_path = os.path.dirname(path) + '/images/desk.jpg'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Apply Harris corner detection
dst = cv2.cornerHarris(img_gray, blockSize = 2, ksize = 3, k = 0.04)

# Spot the detected corners
img_2 = img.copy()
img_2[dst>0.01*dst.max()] = [255, 0, 0]

# Plot the image
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
axs[0].imshow(img); axs[0].axis('off'); axs[0].set_title('Original Image', y = -.2)
axs[1].imshow(img_2); axs[1].axis('off'); axs[1].set_title('Harris Corner Detection', y =-.2)
plt.show()


# Apply Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(img_gray, maxCorners = 50, qualityLevel =0.01,
                                  minDistance = 10)

corners = np.int0(corners)

# Spot the detected corners
img_2 = img.copy()
for i in corners:
    x, y = i.ravel()
    cv2.circle(img_2, center=(x, y), radius = 5, color = 255, thickness = -1)

# Plot the image
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
axs[0].imshow(img); axs[0].axis('off'); axs[0].set_title('Original Image', y = -.2)
axs[1].imshow(img_2); axs[1].axis('off')
axs[1].set_title('Shi-Tomasi Corner Detection', y =-.2)
plt.show()
