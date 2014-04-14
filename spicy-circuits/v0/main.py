
import cv2
import preprocess
import contourFinder

def displayImage(src):
    cv2.imshow('image', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

img = cv2.imread('/Users/Tabish/Desktop/density_window/circuit2Original.jpg', 0)
img = preprocess.convertToBinary(img)
displayImage(img)
contourFinder.contourFinder(img)