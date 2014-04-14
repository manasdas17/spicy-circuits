
import cv2
import numpy as np

#Converts image to binary with option to invert
def convertToBinary(img, invert = 0):
    img = cv2.medianBlur(img, 5)

    if invert == 0:
        ret3, th3 = cv2.threshold(img, 255, 0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif invert == 1:
        ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #Resize image
    img = cv2.resize(th3, (0,0), fx=0.2, fy=0.2)

    return img