import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    sobel_X = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    sobel_Y = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    cv2.imshow('Sobel X', sobel_X)
    cv2.imshow('Sobel Y', sobel_Y) 
    

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    laplacian = np.uint8(laplacian)
    cv2.imshow('laplacian', laplacian)

    cannyedge = cv2.Canny(frame, 120, 120)
    cv2.imshow('canny', cannyedge)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()