import cv2
import numpy as np


"""
Preprocesses an image. Open file -> Resize -> Grayscale -> Thresholding -> Binary

Inputs: 
	img: image path
	size: (width, height)
	threshInv: option to invert when converting to binary

Outputs: Resized image, Grayscale image, Thresholded image, Binary image
"""
def preprocess(img, size, threshInv=False):

	# open file
	img = cv2.imread(img, 1)

	# resize
	img = cv2.resize(img, size)	
	
	# grayscale
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# threshold
	if threshInv == True:
		ret,imthresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	else:
		ret,imthresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)		

	# binary
	imbin = imthresh
	imbin[imbin == 255] = 1

	return img, imgray, imthresh, imbin

