import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

file_path = path.abspath(__file__)
img_path = path.join(path.dirname(file_path), 'images/')

# Import the large image
img = cv2.imread(img_path + '/pine_apple.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
backpacker = cv2.imread(img_path + '/backpacker.jpg')
backpacker = cv2.cvtColor(backpacker, cv2.COLOR_BGR2RGB)
plt.imshow(backpacker)
plt.show()

# Crop the small image and the roi
roi = backpacker[750:1150, 300:500]
img_2 = img[40:440, 80:280]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(roi); axs[1].imshow(img_2)
plt.show()

# Creating the mask for the roi and small image
img_gray = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(img_gray, 254/2+100, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(mask, cmap='gray'); axs[1].imshow(mask_inv, cmap='gray')
plt.show()

# Masking
img_bg = cv2.bitwise_and(roi, roi, mask=mask)
img_fg = cv2.bitwise_and(img_2, img_2, mask=mask_inv)
dst = cv2.add(img_fg, img_bg)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img_bg); axs[1].imshow(img_fg); axs[2].imshow(dst)
plt.show()

# Final output
backpacker[750:1150, 300:500] = dst
plt.imshow(backpacker)
plt.show()
