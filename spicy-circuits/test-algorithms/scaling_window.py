import numpy as np
import cv2
from matplotlib import pyplot as plt

# Algorithm to determine correct component window size, given coordinate of component
# Idea is for user to click on each component, to give us points of interest (POI) to look at.

img = cv2.imread('/Users/Ryan/Desktop/test3.png',0)
ret,binimg = cv2.threshold(img,100,1,cv2.THRESH_BINARY_INV)
print img
POI = (550,660)     # point of interest where user clicks on component
POIx = POI[0]       # x coordinate
POIy = POI[1]       # y coordinate
dwinheight = 1     # window height scales around POI in these increments
dwinwidth = 1      # window width scales around POI in these increments
pxdensity = []
for iteration in xrange(500):
    window = binimg[POIy-dwinheight:POIy+dwinheight, POIx-dwinwidth:POIx+dwinwidth]
    dwinwidth += 1
    dwinheight += 1
    mean = np.mean(window)
    pxdensity.append(mean)
    
    cv2.imshow('image',img[POIy-dwinheight:POIy+dwinheight, POIx-dwinwidth:POIx+dwinwidth])
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

plt.plot(pxdensity)
plt.show()