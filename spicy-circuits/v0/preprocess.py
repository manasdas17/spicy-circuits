
import cv2
import numpy as np

#Converts image to binary with option to invert
def convertToBinary(img, invert = False):
    img = cv2.medianBlur(img, 5)

    if invert == False:
        thresh, imbin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif invert == True:
        thresh, imbin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return imbin