import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Draw a circle by clicking directly on the window.

# Step 1. Define callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, center=(x,y), radius=5, color=(87,184,237), thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, center=(x,y), radius=10, color=(87,184,237), thickness=1)

# Step 2. Call the window
path = os.path.abspath("__file__")
image_path = os.path.join(os.path.dirname(path), 'images/map.PNG')

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.namedWindow(winname='my_desire')
cv2.setMouseCallback('my_desire', draw_circle)

# Step 3. Execution
while True:
    cv2.imshow('my_desire', img)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
