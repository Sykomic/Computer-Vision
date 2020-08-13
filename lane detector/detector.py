import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def do_canny(frame):
    # TODO
    # Converts frames to grayscale because we only need the luminance channel for detecting edges - less computational
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 50, 150)
    return canny

def do_segment(frame):
    # TODO
    height = frame.shape[0]
    polygons = np.array([[(0, height), (800, height), (380, 290)]])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, polygons, 255)
    segment = cv.bitwise_and(frame, mask)

    return segment

def calculate_lines(frame, lines):
    # TODO
    return None

def calculate_coordinates(frame, parameters):
    # TODO
    return None

def visualize_lines(frame, lines):
    # TODO
    return None

# TODO
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("input.mp4")
while (cap.isOpened()):
    ret, frame = cap.read() # ret = a boolean return value from getting the frame,
                            # frame = the current frame being projected in the video.
    canny = do_canny(frame)
    segment = do_segment(canny)

    cv.imshow('input', segment)
    if cv.waitKey(10) & 0xFF == ord('q'): # Frames are read by intervals of 10 milliseconds.
        break                            # The programse breaks out of the while loop when the user presses the 'q' key

# Frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
