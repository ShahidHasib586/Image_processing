import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    # Roberts operator
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv2.filter2D(frame, cv2.CV_16S, kernelx)
    y = cv2.filter2D(frame, cv2.CV_16S, kernely)

    # Turn uint8, image fusion
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow('Roberts', Roberts)

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