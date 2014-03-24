import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import cos, tan, radians
import time
from numba import jit
from peak_detect import *
import random

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

def _lineTraceGenerator(degrees, step):
	i = step
	if (0 <= (degrees % 180) <= 45) or (135 <= (degrees % 180) <= 180):
		k = abs(tan(radians(degrees))) if 0 <= degrees <= 180 else -1 * abs(tan(radians(degrees)))
		while True:
			if 90 < degrees < 270:
				yield (-i, int(round(i * k)))
				i += step
			else:
				yield (i, int(round(i * k)))
				i += step
	else:
		k = cos(radians(degrees))
		while True:
			if 180 < degrees < 360:
				yield (int(round(i * k)), -i)
				i += step
			else:
				yield (int(round(i * k)), i)
				i += step

@jit
def _windowOnLine(window, direction):	
	wrongColumns = 0
	if direction == 1:	
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
	else:
		numRows = len(window[1])
		currentRow = numRows / 2
		while True:
			if (wrongColumns >= numRows): 			
				return False
			if (window[0, currentRow] == 1 
				and window[1, currentRow] == 1):
				return True
			else:
				currentRow = (currentRow + 1) % numRows
				wrongColumns += 1

 ########################## scanLine ###########################
# Description: 	window moves along straight path and stops after	
#				there are no black pixels in front of it. Use dst  	
#				argument to visualize algorithm. 					
# 																	
# Inputs:	img:		binary image to scan [numpy array] 			
#			startPos:	starting position (pxX,pxY) 				
#			angle:		angle for window to move along in degrees 	
# Optional:	winHeight:	height of moving window in px 				
#			step:		increments of moving window 				
#			dst:		image to display results on, and step  		
#						through code. Note - when image appears,  	
#						hit any key to view next image.				 
# 																	
# Output:	coordinate where window fell off line (pxX,pxY) 		

def scanLine(img, startPos, angle, winHeight=6, step=2, dst=None):	
	traceGenerator = _lineTraceGenerator(angle, step)
	while angle < 0:
		angle += 360
	if (0 <= (angle % 180) <= 45) or (135 <= (angle % 180) <= 180):
		winLeft, winRight = startPos[0]-(winHeight/2), startPos[0]+(winHeight/2)
		winBack, winFront = startPos[1], startPos[1] + 1
		window = img[winLeft:winRight, winBack:winBack+2]
		direction = 1 	# 1 = right/left
		while _windowOnLine(window, direction):	
			if dst != None:
				imgUpdate = dst.copy()
				cv2.rectangle(imgUpdate, (winBack, winLeft), (winFront, winRight), 120, 2)
				displayImage(imgUpdate)
			stepCoords = next(traceGenerator)
			winLeft = startPos[0]-(winHeight/2) - stepCoords[1]
			winRight = startPos[0] + (winHeight/2) - stepCoords[1]
			window = img[winLeft:winRight, winBack:winBack+2]
			winBack, winFront = startPos[1] + stepCoords[0], startPos[1] + stepCoords[0]
		if 90 <= angle <= 270:
			compLeftNode = (int(np.mean((winLeft, winRight))), winBack + (step * 4))
		else:
			compLeftNode = (int(np.mean((winLeft, winRight))), winBack - (step * 4))
		if dst != None:	
			imgUpdate = dst.copy()
			cv2.rectangle(imgUpdate, (compLeftNode[1]-step,compLeftNode[0]-9), (compLeftNode[1]+9,compLeftNode[0]+9), 100, 10)
			displayImage(imgUpdate)
		return compLeftNode

	else:
		winLeft, winRight = startPos[1]-(winHeight/2), startPos[1]+(winHeight/2)
		winBack, winFront = startPos[0], startPos[0] + 1
		window = img[winBack:winBack+2, winLeft:winRight]
		direction = 0 	# 0 = up/down
		while _windowOnLine(window, direction):	
			if dst != None:
				imgUpdate = dst.copy()
				cv2.rectangle(imgUpdate, (winLeft, winFront), (winRight, winBack), 120, 2)
				displayImage(imgUpdate) 
			stepCoords = next(traceGenerator)
			winLeft = startPos[1]-(winHeight/2) - stepCoords[0]
			winRight = startPos[1] + (winHeight/2) - stepCoords[0]
			window = img[winBack:winBack+2, winLeft:winRight]
			winBack, winFront = startPos[0] - stepCoords[1], startPos[0] - stepCoords[1]
		if 0 <= angle <= 180: 
			compLeftNode = (winBack + (step * 4), int(np.mean((winLeft, winRight))))
		else:
			compLeftNode = (winBack - (step * 4), int(np.mean((winLeft, winRight))))
		if dst != None:	
			imgUpdate = dst.copy()
			cv2.rectangle(imgUpdate, (compLeftNode[1]-9,compLeftNode[0]-9), (compLeftNode[1]+9,compLeftNode[0]+9), 100, 10)
			displayImage(imgUpdate)
		return compLeftNode

def distance(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2

drawImg = cv2.imread('/Users/Ryan/Desktop/test3.png',1)
img = cv2.imread('/Users/Ryan/Desktop/test3.png',0) # get grayscale image
ret,binimg = cv2.threshold(img,100,1,cv2.THRESH_BINARY_INV)
imgHeight, imgWidth = len(binimg), len(binimg[0])

def detectNodes():	
	while True:
		randX, randY = random.randint(0,imgWidth - 1), random.randint(0,imgHeight - 1)
		if binimg[randY, randX] == 1:
			break
	startPoint = (randY,randX)	
	startPoint = (randY,randX)	
	toCheck = [startPoint]
	checked = []
	found = []
	while toCheck:
		origin = toCheck[-1]
		checked.append(origin)
		nodes = []
		for x in range(0,370):	
			nodes.append(scanLine(binimg, origin, x, winHeight = 10))
		windowDist = [distance(x,origin) for x in nodes]
		peaks = peakdetect(windowDist, lookahead = 6, delta = 100)[0]
		if len(peaks) == 0:
			break
		peakX, peakY = zip(*peaks)
		peakX = [e for e in peakX]
		toCheck.pop()
		for e in peakX:
			node = nodes[e]
			tooClose = False
			for j in checked:
				if distance(j, node) < 600:
					tooClose = True
					break
			if node not in checked and tooClose == False:	
				toCheck.append(node)
				found.append(node)	
		# plt.plot(windowDist)
		# plt.plot(peakX, peakY, 'ro')
		# plt.show()		
		imgUpdate = drawImg.copy()		
		for x in nodes:
			cv2.rectangle(imgUpdate, (x[1]-3,x[0]-3), (x[1]+3,x[0]+3), (255,0,0), 1)	
		for node in found:
			cv2.rectangle(imgUpdate, (node[1]-8,node[0]-8), (node[1]+8,node[0]+8), (50,255,50), -1)
		for x in nodes:
			cv2.rectangle(imgUpdate, (origin[1]-7,origin[0]-7), (origin[1]+7,origin[0]+7), (0,0,255), 2)
		displayImage(imgUpdate)
	for node in found:
		cv2.rectangle(imgUpdate, (node[1]-8,node[0]-8), (node[1]+8,node[0]+8), (50,255,50), -1)
	displayImage(imgUpdate)
# start = time.clock()
# for x in xrange(100):	
# 	start1 = time.clock()
# 	detectNodes()
# 	print x + 1, time.clock() - start1
# print time.clock() - start
detectNodes()
