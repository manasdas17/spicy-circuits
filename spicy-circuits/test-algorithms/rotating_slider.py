import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# checks whether or not window is on line
def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

def windowOnLine(window):	
	wrongColumnsChecked = 0
	columnHeight = len(window)
	currentRow = 0
	while True:
		if (wrongColumnsChecked >= columnHeight): 			
			return False
		if (window[currentRow, 0] == 1 
			and window[currentRow, 1] == 1):
			return True
		else:
			currentRow += 1
			wrongColumnsChecked += 1

def scanLine(angle=0):
	img = cv2.imread('/Users/Ryan/Desktop/test3.png',0) # get grayscale image
	ret,binimg = cv2.threshold(img,100,1,cv2.THRESH_BINARY_INV)
	winUp, winDown, winLeft, winRight = 620, 660, 440, 441 # starting window location
	while windowOnLine(binimg[winUp:winDown, winLeft:winLeft+2]):
		# imgUpdate = img.copy()
		# cv2.rectangle(imgUpdate, (winLeft, winUp), (winRight, winDown), 120, 2)
		# displayImage(imgUpdate) 
		winLeft, winRight = winLeft + 5, winRight + 5 # move window right
	compLeftNode = (np.mean((winUp, winDown)), winLeft)
	cv2.rectangle(img, (winLeft-14,int(compLeftNode[0])-9), (winLeft+4,int(compLeftNode[0])+9), 100, -1)
	# displayImage(img)

start = time.clock()
for x in xrange(100):
	scanLine()
print time.clock()-start


