# video show in open cv
import cv2
import numpy as np
import os

file_path = os.path.join(os.getcwd(), '1.mp4')
cap = cv2.VideoCapture(file_path)

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        cv2.imshow('Frame', frame)
        cv2.resize(frame,(320,110),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()

cv2.destroyAllWindows()