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
			compLeftNode = (int(np.mean((winLeft, winRight))), winBack + (step * 5))
		else:
			compLeftNode = (int(np.mean((winLeft, winRight))), winBack - (step * 5))
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
			compLeftNode = (winBack + (step * 5), int(np.mean((winLeft, winRight))))
		else:
			compLeftNode = (winBack - (step * 5), int(np.mean((winLeft, winRight))))
		if dst != None:	
			imgUpdate = dst.copy()
			cv2.rectangle(imgUpdate, (compLeftNode[1]-9,compLeftNode[0]-9), (compLeftNode[1]+9,compLeftNode[0]+9), 100, 10)
			displayImage(imgUpdate)
		return compLeftNode

def distance(p0, p1):
    return ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)**.5

# generate list of coordinates of black pixels in tuple form.
@jit
def getBlackPx(img):
	blackPxToCheck = []
	for row in enumerate(img):
		for px in enumerate(row[1]):
			if px[1] == 1:
				blackPxToCheck.append((row[0], px[0]))
	return blackPxToCheck

def detectNodes(distThresh=20, histogram=False, periodicUpdate=False):	
	# generate random starting point on black pixel 
	while True:
		randX, randY = random.randint(0,imgWidth - 1), random.randint(0,imgHeight - 1)
		if binimg[randY, randX] == 1:
			break
	startPoint = (randY,randX)	
	startPoint = (randY,randX)	
	# list of points to check, starting with starting point. 
	# nodes found appended to list. Nodes checked popped from list
	toCheck = [startPoint]
	# if a node is in this list, we already checked it. Don't check again.
	checked = []
	# coordinates of nodes found
	found = []
	# done when no more nodes to check
	while toCheck:
		origin = toCheck[-1]
		checked.append(toCheck.pop())
		# list of possible node coordinates, indexed by angle
		possibleNodes = []
		# run scanline algorithm for all angles in circle
		for x in range(0,370):
			possibleNodes.append(scanLine(binimg, origin, x, winHeight = 8))
		# list of distances from where window fell off line to origin
		windowDist = [distance(node,origin) for node in possibleNodes]
		# nodes we're interested in are where the window travelled farthest
		peaks = peakdetect(windowDist, lookahead = 5, delta = 10)[0]
		if len(peaks) == 0:
			break
		peakAngles, peakDistances = zip(*peaks)
		# make sure potential node isn't too close to already found nodes
		# to avoid clusters of nodes
		for peakNode in [possibleNodes[angle] for angle in peakAngles]:
			tooClose = False	
			for foundNode in found:
				if distance(foundNode, peakNode) < distThresh:
					tooClose = True
					break
			if not tooClose:	
				toCheck.append(peakNode)
				found.append(peakNode)	
		if histogram == True:
			plt.plot(windowDist)
			plt.plot(peakAngles, peakDistances, 'ro')
			plt.show()			
		if periodicUpdate == True:	
			imgUpdate = drawImg.copy()				
			for x in possibleNodes:
				cv2.rectangle(imgUpdate, (x[1]-3,x[0]-3), (x[1]+3,x[0]+3), (255,0,0), 1)	
			for node in found:
				cv2.rectangle(imgUpdate, (node[1]-8,node[0]-8), (node[1]+8,node[0]+8), (50,255,50), -1)
			for x in possibleNodes:
				cv2.rectangle(imgUpdate, (origin[1]-7,origin[0]-7), (origin[1]+7,origin[0]+7), (0,0,255), 2)
			displayImage(imgUpdate)
	imgUpdate = drawImg.copy()	
	for node in found:
		cv2.rectangle(imgUpdate, (node[1]-8,node[0]-8), (node[1]+8,node[0]+8), (50,255,50), -1)
	displayImage(imgUpdate)

drawImg = cv2.imread('/Users/Ryan/Desktop/test3.png',1)
img = cv2.imread('/Users/Ryan/Desktop/test3.png',0) # get grayscale image
ret,binimg = cv2.threshold(img,100,1,cv2.THRESH_BINARY_INV)
#print getBlackPx(binimg)
imgHeight, imgWidth = len(binimg), len(binimg[0])
detectNodes(periodicUpdate=True)











