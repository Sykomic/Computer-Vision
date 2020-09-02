import cv2
import matplotlib.pyplot as plt
import os

coding_path = os.path.abspath('thresholding.py')
dir_path = os.path.dirname(coding_path)
img = cv2.imread(dir_path + '/images/gradation.png')

# Thresholding (threshold: 127, maxval: 255) 255 -> white, 0 -> black
_, thresh_0 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, thresh_1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh_2 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, thresh_3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
_, thresh_4 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

# Plot the images
images = [img, thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]

fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (13, 8))
fig.suptitle('Different Types of Thresholding')

for ind, p in enumerate(images):
    ax = axs[ind//3, ind%3]
    ax.imshow(p)
plt.show()


# Convert the image to grayscale
img = cv2.imread(dir_path + '/images/text.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding
_, thresh_binary = cv2.threshold(img, thresh = 127, maxval = 255,
type = cv2.THRESH_BINARY)
adap_mean_2 = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 7, 2)
adap_mean_2_inv = cv2.adaptiveThreshold(img, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 7, 2)
adap_mean_8 = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 7, 8)
adap_gaussian_8 = cv2.adaptiveThreshold(img, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 7, 8)

# Plot the images
images = [img, thresh_binary, adap_mean_2, adap_mean_2_inv, adap_mean_8,
adap_gaussian_8]
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
fig.suptitle('Different types of Adaptive Threshold')
for ind, p in enumerate(images):
    ax = axs[ind%2, ind//2]
    ax.imshow(p, cmap='gray')
    ax.axis('off')
plt.show()


# Gradient
# Apply gradient filtering
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Plot the images
images = [sobel_x, sobel_y, blended, laplacian]
plt.figure(figsize=(20, 8))
plt.title('Gradient Filtering')
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.show()
