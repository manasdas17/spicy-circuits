import numpy as np 
import cv2
from skimage.morphology import skeletonize

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

# open image, gray image, and binary image
img = cv2.imread('/Users/Ryan/Desktop/test6.png',1)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,imbin = cv2.threshold(imgray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# skeletonizes image
imskel = skeletonize(imbin).astype(np.uint8)

# generates contour vectors
contours, hierarchy = cv2.findContours(imbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# approximates line segments from contours
segments = [cv2.approxPolyDP(contour,1,True) for contour in contours]
print len(segments)

cv2.drawContours(img,segments,-1,(0,255,0),1)

# draw contours on image
#for e in segments:
#	cv2.drawContours(img,e,-1,(0,255,0),3)

displayImage(img)