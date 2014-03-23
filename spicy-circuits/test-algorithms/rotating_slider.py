import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from numba import jit

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

@jit
def windowOnLine(window):	
	wrongColumns = 0
	numRows = len(window)
	currentRow = numRows / 2
	while True:
		if (wrongColumns >= numRows): 			
			return False
		if (window[currentRow, 0] == 1 
			and window[currentRow, 1] == 1):
			return True
		else:
			currentRow = (currentRow + 1) % numRows
			wrongColumns += 1

def scanLine(img, winHeight, startPos, angle=0, step=2, dst=None):	
	winUp, winDown = startPos[0]-(winHeight/2), startPos[0]+(winHeight/2)
	winLeft, winRight = startPos[1], startPos[1] + 1
	window = img[winUp:winDown, winLeft:winLeft+2]
	while windowOnLine(window):
		if dst != None:
			imgUpdate = dst.copy()
			cv2.rectangle(imgUpdate, (winLeft, winUp), (winRight, winDown), 120, 2)
			displayImage(imgUpdate) 
		winLeft, winRight = winLeft + step, winRight + step
		window = img[winUp:winDown, winLeft:winLeft+2]
	compLeftNode = (np.mean((winUp, winDown)), winLeft)
	if dst != None:	
		imgUpdate = dst.copy()
		cv2.rectangle(imgUpdate, (winLeft-14,int(compLeftNode[0])-9), (winLeft+4,int(compLeftNode[0])+9), 100, -1)
		displayImage(imgUpdate)
	return compLeftNode


