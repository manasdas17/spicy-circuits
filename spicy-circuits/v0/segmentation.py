import numpy as np 
import cv2
from skimage.morphology import skeletonize, disk

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

"""
Returns True if rectangle is enclosed by any rectangles in reclist.

Rectangles are of the form ((x1,y1),(x2,y2)), where (x1,y1) is the
top left corner, and (x2,y2) is the lower right. reclist is a list 
of these rectangles.
"""

def _inRec(rec, reclist):
	for each in reclist:
		if (rec[0][0] > each[0][0] and rec[0][1] > each[0][1] 
		and rec[1][0] < each[1][0] and rec[1][1] < each[1][1]):
			return True

	else:
		return False

"""
Finds rectangular regions of interest (corners, components, nodes) using
moving pixel density window. Implemented with convolution kernel for speed.
Works best on smaller images.

Inputs:
	img - binary image (0 and 1)
	windowSize - length of square window

Outputs:
	list of rectangles corresponding to each found component
"""
def findROI(img, windowSize=15):
	
	# skeletonize image to normalize pixel density across image
	imskel = skeletonize(img).astype(np.uint8)

	# convolution kernel used to simulate moving window
	# heavily weighted anchor point to simulate movement along line
	radius = windowSize / 2
	kernel = np.ones((windowSize, windowSize), np.uint8)
	anchor = radius ** 2
	kernel[radius, radius] = anchor

	# moving pixel density window with kernel
	img = cv2.filter2D(imskel, -1, kernel)
	
	# threshold is (diameter + 1) to capture everything excluding lines
	thresh = windowSize + anchor - 1

	# binarize image to threshold
	thresh, imbin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

	# overlap pixels in pixel dense regions with dilation
	kernel = disk(3)
	dilation = cv2.dilate(imbin, kernel, iterations=1)

	# draw contour around each pixel dense region
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# get minimum enclosing rectangle around each region
	regions = [cv2.boundingRect(contour) for contour in contours]
	regions = [((rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3])) for rec in regions]

	# remove regions enclosed by other regions
	regions = [rec for rec in regions if not _inRec(rec, regions)]

	return regions



img = cv2.imread('/Users/Ryan/Desktop/test_images/test10.png',1)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,imbin = cv2.threshold(imgray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# get regions of interest
regions = findROI(imbin)

# display regions
for region in regions:
	cv2.rectangle(img,region[0],region[1],(0,255,0),1)

displayImage(img)




