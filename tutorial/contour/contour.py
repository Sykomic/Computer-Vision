import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = os.path.abspath(__file__)
img_path = os.path.join(os.path.dirname(file_path), 'images/pine_apple.jpg')

# Load the image
img = plt.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Blurring for removing the noise
img_blur = cv2.bilateralFilter(img, d=7, sigmaSpace=75, sigmaColor=75)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Apply the thresholding
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a/2+60, a, cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap='gray')
plt.show()

# Find the contour of the figure
# images, contours, hierarchy --> contours, hierarchy (cv2 version is changed)
contours, hierarchy = cv2.findContours(image=thresh,
                                              mode=cv2.RETR_TREE,
                                              method=cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours
contours = sorted(contours, key = cv2.contourArea, reverse=True)

# Draw the contour
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx=-1, color=(255, 0, 0),
                         thickness=2)
plt.imshow(img_copy)
plt.show()


# The first order of the contours
c_0 = contours[0]

# image moment
M = cv2.moments(c_0)
print(M.keys())

# The area of contours
print('1st Contour Area :', cv2.contourArea(contours[0]))
print('2nd Contour Area :', cv2.contourArea(contours[1]))
print('3rd Contour Area :', cv2.contourArea(contours[2]))

# The arc length of contours
print(cv2.arcLength(contours[0], closed=True))
print(cv2.arcLength(contours[0], closed=False))

# The centroid point
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

# The extreme points
l_m = tuple(c_0[c_0[:, :, 0].argmin()][0])
r_m = tuple(c_0[c_0[:, :, 0].argmax()][0])
t_m = tuple(c_0[c_0[:, :, 1].argmin()][0])
b_m = tuple(c_0[c_0[:, :, 1].argmax()][0])

pst = [l_m, r_m, t_m, b_m]
xcor = [p[0] for p in pst]
ycor = [p[1] for p in pst]

# Plot the points
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axs[0].imshow(thresh, cmap='gray'); axs[0].scatter([cx], [cy], c='b', s=50)
axs[1].imshow(thresh, cmap='gray'); axs[1].scatter(xcor, ycor, c='b', s=50)
plt.show()

# The first order of the contours
c_0 = contours[0]

# Get the 4 points of the bounding rectangle
x, y, w, h = cv2.boundingRect(c_0)

# Draw a straight rectangle with the points
img_copy = img.copy()
img_box = cv2.rectangle(img_copy, (x, y), (x+w, y+h), color=(255,0,0), thickness=2)


# Get the 4 points of the bounding rectangle w/ the min area
rect = cv2.minAreaRect(c_0)
box = cv2.boxPoints(rect)
box = box.astype('int')

# Draw a contour w/ the points
img_copy = img.copy()
img_box_2 = cv2.drawContours(img_copy, contours=[box], contourIdx = -1,
                            color=(255,0,0), thickness=2)

# Detect the convex contour
hull = cv2.convexHull(c_0)
img_copy = img.copy()
img_hull = cv2.drawContours(img_copy, contours=[hull], contourIdx=-1,
                            color=(255,0,0), thickness=2)

fig, axs = plt.subplots(1, 3, figsize=(10, 8))
axs[0].imshow(img_box)
axs[1].imshow(img_box_2)
axs[2].imshow(img_hull)
plt.show()
