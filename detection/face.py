import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.path.abspath('face.py')
img_path = os.path.dirname(path) + '/images/captin_marvel.jpg'

cap_mavl = cv2.imread(img_path)

# Find the region of interest
roi = cap_mavl[50:350, 200:550]
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
plt.imshow(roi, cmap='gray')
plt.show()

user_path = os.path.abspath('../../../')
cascade_path = os.path.join(user_path, 'anaconda3/lib/python3.7/site-packages/cv2/\
data/haarcascade_frontalface_default.xml')

# Load Cascade filter
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create the face detecting function
def detect_face(img):
    img_copy = img.copy()
    face_rects = face_cascade.detectMultiScale(img_copy, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 255), 3)
    return img_copy

# Detect the face
roi_detected = detect_face(roi)
plt.imshow(roi_detected, cmap='gray')
plt.axis('off')
plt.show()


# Load the image file and convert the color mode
avg_path = os.path.join(os.path.dirname(path), 'images/avengers.jpg')
avengers = cv2.imread(avg_path)
avengers = cv2.cvtColor(avengers, cv2.COLOR_BGR2GRAY)

# Detect the face and plot the result
detected_avengers = detect_face(avengers)
plt.imshow(detected_avengers, cmap='gray')
plt.axis('off')
plt.show()


# Detecting your face. Press 'ESC' to quit.
cap = cv2.VideoCapture(0) # 0 refers to the first camera. If you have a second camera, put 1, and so on.
while True:
    ret, frame = cap.read(0)
    frame = detect_face(frame)
    cv2.imshow('Video Face Detection', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
