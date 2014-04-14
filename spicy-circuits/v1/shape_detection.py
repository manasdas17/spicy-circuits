import cv2
import cv2.cv
import cv
import numpy as np
from scipy.spatial import distance

'''
Uses for Contours:

    Get width of line
    Shape Detection -> circles, plus sign, triangle, rectangles
    Seperation of meshes -> Run algorithm on each mesh and combine result
    Isolation of numbers and symbols ->

'''

def displayImage(src):
    cv2.imshow('image', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

#Finds contours by first applying canny edge detection and returns two contours
def contourFinder(img):
    thresh = 100
    edges = cv2.Canny(img, thresh, thresh * 2) #Canny Edge detection
    drawing = np.zeros(img.shape, np.uint8) #Image to draw the contours
    
    #Return all coordinates of contour. No approximation.
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for each in contours:
        cv2.drawContours(drawing, [each], 0, (255, 255, 255))
        cimg = cv2.cvtColor(drawing,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(drawing,cv.CV_HOUGH_GRADIENT,1,20,
                            param1=50,param2=24,minRadius=0,maxRadius=0)
        
        if circles != None:
            for i in circles[0, :]:
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) 
            displayImage(cimg)

        drawing = np.zeros(img.shape, np.uint8)

    return drawing

#Returns width of line
def getLineWidth():
    pass


