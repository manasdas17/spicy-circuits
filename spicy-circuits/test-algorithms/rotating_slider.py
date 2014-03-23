import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import cos, tan, radians
import time
from numba import jit

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

def lineTraceGenerator(degrees, step):
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
def windowOnLine(window, direction):	
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

 ########################### scanLine ##############################
# Description: 	window moves along straight path and stops after	#
#				there are no black pixels in front of it. Use dst  	#
#				argument to visualize algorithm. 					#
# 																	#
# Inputs:	img:		binary image to scan [numpy array] 			#
#			startPos:	starting position (pxX,pxY) 				#
#			angle:		angle for window to move along in degrees 	#
# Optional:	winHeight:	height of moving window in px 				#
#			step:		increments of moving window 				#
#			dst:		image to display results on, and step  		#
#						through code. Note - when image appears,  	#
#						hit any key to view next image.				# 
# 																	#
# Output:	coordinate where window fell off line (pxX,pxY) 		#
 ###################################################################

def scanLine(img, startPos, angle, winHeight=5, step=2, dst=None):	
	traceGenerator = lineTraceGenerator(angle, step)
	while angle < 0:
		angle += 360
	if (0 <= (angle % 180) <= 45) or (135 <= (angle % 180) <= 180):
		winUp, winDown = startPos[0]-(winHeight/2), startPos[0]+(winHeight/2)
		winBack, winFront = startPos[1], startPos[1] + 1
		window = img[winUp:winDown, winBack:winBack+2]
		direction = 1 	# 1 = right/left
		while windowOnLine(window, direction):	
			if dst != None:
				imgUpdate = dst.copy()
				cv2.rectangle(imgUpdate, (winBack, winUp), (winFront, winDown), 120, 2)
				displayImage(imgUpdate)
			stepCoords = next(traceGenerator)
			winUp = startPos[0]-(winHeight/2) - stepCoords[1]
			winDown = startPos[0] + (winHeight/2) - stepCoords[1]
			window = img[winUp:winDown, winBack:winBack+2]
			winBack, winFront = startPos[1] + stepCoords[0], startPos[1] + stepCoords[0]
		compLeftNode = (int(np.mean((winUp, winDown))), winBack - step)
		if dst != None:	
			imgUpdate = dst.copy()
			cv2.rectangle(imgUpdate, (compLeftNode[1]-step,compLeftNode[0]-9), (compLeftNode[1]+9,compLeftNode[0]+9), 100, 10)
			displayImage(imgUpdate)
		return compLeftNode

	else:
		winUp, winDown = startPos[1]-(winHeight/2), startPos[1]+(winHeight/2)
		winBack, winFront = startPos[0], startPos[0] + 1
		window = img[winBack:winBack+2, winUp:winDown]
		direction = 0 	# 0 = up/down
		while windowOnLine(window, direction):	
			if dst != None:
				imgUpdate = dst.copy()
				cv2.rectangle(imgUpdate, (winUp, winFront), (winDown, winBack), 120, 2)
				displayImage(imgUpdate) 
			stepCoords = next(traceGenerator)
			winUp = startPos[1]-(winHeight/2) - stepCoords[0]
			winDown = startPos[1] + (winHeight/2) - stepCoords[0]
			window = img[winBack:winBack+2, winUp:winDown]
			winBack, winFront = startPos[0] - stepCoords[1], startPos[0] - stepCoords[1]
		compLeftNode = (winBack + step, int(np.mean((winUp, winDown))))
		if dst != None:	
			imgUpdate = dst.copy()
			cv2.rectangle(imgUpdate, (compLeftNode[1]-9,compLeftNode[0]-9), (compLeftNode[1]+9,compLeftNode[0]+9), 100, 10)
			displayImage(imgUpdate)
		return compLeftNode


