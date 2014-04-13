import numpy as np 
import cv2
from skimage.morphology import skeletonize

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

img = cv2.imread('/Users/Ryan/Desktop/test6.png',1)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,imbin = cv2.threshold(imgray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


imskel = skeletonize(imbin).astype(np.uint8)

contours, hierarchy = cv2.findContours(imskel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
	approxCurve = cv2.approxPolyDP(contour,3,True)
	cv2.drawContours(img,approxCurve,-1,(0,255,0),3)

displayImage(img)