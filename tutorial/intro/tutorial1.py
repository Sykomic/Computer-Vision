import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
notebook_path = os.path.abspath("Untitled.ipynb")
image_path = os.path.join(os.path.dirname(notebook_path), 'images/burano.jpg')

img = cv2.imread(image_path)
plt.imshow(img)
plt.title('BGR Image')
plt.show()

# Convert the image into RGB from BGR (OpenCV deafult)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('RGB Image')
plt.show()

"""
A color model: a system for creating a full range of colors using the primary colors.
Additive color models: uses light to represent colors in computer screens. (RGB)
Subtractive: uses inks to print those digital images on papers. (CMYK) cyan, magenta, yellow and black.
"""

# Convert the image into gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray') # w/o cmap, it's greenish.
plt.title('Grayscale')
plt.show()


"""
RGB images are stacked by three channels: R, G, and B.
Let's plot them one by one, and understand how the color channels are structured.
"""

# Plot the three channels of the image
fig, axs = plt.subplots(nrows=1, ncols=3)

for i in range(0, 3):
    ax = axs[i]
    ax.imshow(img_rgb[:, :, i], cmap='gray')
axs[0].set_title('R channel')
axs[1].set_title('G channel')
axs[2].set_title('B channel')

plt.show()

"""
In grayscale mode, the higher the saturation of red colors, the whiter the color becomes in each channel picture.
HSV and HLS: three-dimensional representation. more similar to the way of human representation.
HSV: hue(the actual colors), saturation, value
HLS: hue, saturation, lightness
"""

# Transform the image into HSV and HLS models
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# Plot the converted images
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(img_hsv)
axs[0].set_title('HSV')
axs[1].imshow(img_hls)
axs[1].set_title('HLS')
plt.show()

# Drawing a rectangle on 'the wall of love' img.
# need to give the coordinates values for the upper left corner and the lower right corner.

img_wallP = os.path.join(os.path.dirname(notebook_path), 'images/wall.jpg')
img_wall = cv2.imread(img_wallP)
img_wall = cv2.cvtColor(img_wall, cv2.COLOR_BGR2RGB)
plt.imshow(img_wall)
plt.show()

# Copy the image
wall_copy = img_wall.copy()

# Draw a rectangle
cv2.rectangle(wall_copy, pt1=(800, 470), pt2=(980, 530), color=(255,0,0), thickness=5)
plt.imshow(wall_copy)
plt.show()

# Draw a circle. Need to specify the center and the length of its radius.

cv2.circle(wall_copy, center=(950,50), radius=50, color=(0,128,0), thickness=5)
plt.imshow(wall_copy)
plt.show()

# Put text. Need to specify the upper left corner.
cv2.putText(wall_copy, text='The Wall of Love', org=(250, 250), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
plt.imshow(wall_copy)
plt.show()

