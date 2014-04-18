import numpy as np 
import cv2

def displayImage(src):
	cv2.imshow('image',src)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

"""
Generates classifier for components by averaging hu moments of sample images

Input: list of file paths
Output: list of moments
"""
def classifierGenerator(samples):

	# generate list of hu moments for each image in images
	hu = []
	for sample in samples:

		# process image
		img = cv2.imread(sample,1)
		imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret,imbin = cv2.threshold(imgray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		
		# resize image
		imbin = cv2.resize(imbin, (50,50))

		# generate moments
		moments = cv2.moments(imbin)
		huMoments = cv2.HuMoments(moments)
		huMoments = [e[0] for e in huMoments]
		hu.append(huMoments)

	# average moments of each image 
	avgHu = []
	for i in xrange(0,7):
		moment = sum([e[i] for e in hu])	
		avgHu.append(moment)
	
	return avgHu


"""
Generates a value corresponding to the match between an unknown object and a 
particular component class using hu moments.

Inputs:
	object: binary image of object to test
	classifier: hu classifier of component class, generated with classifierGenerator

Output: 
	value corresponding to match
"""
def componentMatch(object, classifier):
	
	# generate moments for object
	objMoments = cv2.moments(object)
	objHu = cv2.HuMoments(objMoments)
	objHu = [e[0] for e in objHu]

	# compare moments to component moments
	score = 0
	for i in xrange(0,7):
		score += abs(objHu[i] - classifier[i])

	return score

resistorTrainingSamples = ['/Users/Ryan/Desktop/test_images/Components/Resistors/sample' 
		+ str(i) + '.png' for i in range(2,4)]

resistorSamples = ['/Users/Ryan/Desktop/test_images/Components/Resistors/sample' 
		+ str(i) + '.png' for i in range(1,6)]

nonResistorSamples = ['/Users/Ryan/Desktop/test_images/Components/NonResistors/sample' 
		+ str(i) + '.png' for i in range(1,6)]		

resistorClass = classifierGenerator(resistorTrainingSamples)

print
print "-----Resistor samples------"

for sample in resistorSamples:
	img = cv2.imread(sample,1)
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,imbin = cv2.threshold(imgray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	imbin = cv2.resize(imbin, (50,50))
	print componentMatch(imbin, resistorClass)

print
print "----NonResistor samples----"

for sample in nonResistorSamples:
	img = cv2.imread(sample,1)
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,imbin = cv2.threshold(imgray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	imbin = cv2.resize(imbin, (50,50))
	print componentMatch(imbin, resistorClass)

print

