import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
notebook_path = os.path.abspath("Untitled.ipynb")
image_path = os.path.join(os.path.dirname(notebook_path), 'images/burano.jpg')

img = cv2.imread(image_path)
plt.imshow(img)
plt.show()
