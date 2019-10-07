#
#Summer Project--Particle Shape Analysis

#Import extensive modules

import math
from PIL import Image
import cv2
import numpy as np
from imutils import perspective
from imutils import contours

import imutils
import glob
from itertools import chain

#import and transform the image
image = cv2.imread("Img1.png")

resized = imutils.resize(image,width=300)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find and sort the contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

areas=[]
perimeter=[]
for i in cnts:
           
    #obtain particle area
    areas.append(cv2.contourArea(i))
    #obtain particle perimeter
    perimeter.append(round(cv2.arcLength(i,True),2))
        

reg=[]
for i in range(len(areas)):
        rd= 4*(math.pi)*areas[i]/(int(perimeter[i]))**2
        reg.append(round(rd,2))
print("the area of cnt:", areas)

print("the perimeter of cnt",perimeter)
print("the regularity of each particle:",reg)

##END##
