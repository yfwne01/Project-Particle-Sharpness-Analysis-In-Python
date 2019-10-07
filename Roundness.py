#
#Summer Project--Particle Shape Analysis

#Import extensive modules

import math
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.optimize
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

import imutils
import glob
from itertools import chain

#define the auto canny function to detect the edges
def auto_canny(image, sigma=0.3):
	# compute the median of the single channel pixel intensities
	med = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * med))
	upper = int(min(255, (1.0 + sigma) * med))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

        
class ShapeDetector:
        def __init__(self):
                pass
        def detect(self,c):
                #initialize the shape nae and approximate the contour
                shape="unidentified"
                peri=cv2.arcLength(c,True)
                # c is the contour value
                # normally the range of 1-5% of the original contour perimeter (#0.04 here)
                approx=cv2.approxPolyDP(c,0.04*peri,True)

                #if the shae is a triangle, it will have 3 vertices
                if len(approx)==3:
                        shape = "triangle"
                #if the shape has 4 vertices, it is either a square or a rectangle
                elif len(approx)==4:
                        (x,y,w,h)=cv2.boundingRect(approx)
                        ratio=w/ float(h)

                        shape ="square" if ratio >=0.95 and ratio <=1.05 else "rectangle"
                elif len(approx)==5:
                        shape ="pentagon"
                elif len(approx)==6:
                        shape = "hexagon"
               
                else:
                        shape="cirlce"
                
                return shape, approx

#define dim function to obtain the dimension for each particle
def dim():
    
        #import and transform the image
        image = cv2.imread("Img19.png")
        #print(image.load())
        resized = imutils.resize(image,width=100)
        ratio = image.shape[0] / float(resized.shape[0])
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        Gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(Gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts1 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)
        sd = ShapeDetector()
        
        Cnts1=np.array(cnts1).tolist()
        Cnts1=list(chain(*Cnts1))
        CNT=np.array(Cnts1).tolist()
        CNT=list(chain(*CNT))
        
        cont1, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt1 = cont1[0]
        
        #obtain the radius of the particle
        (x,y),radius = cv2.minEnclosingCircle(cnt1)
        center = (int(x),int(y))
        radius = int(radius)
        img = cv2.circle(image,center,radius,(0,255,0),2)
        imgplot = plt.imshow(img)
        print("the radius of the enclosed circle is:", radius)
        print("X:",round(x,2),"Y:",round(y,2))
        #print(len(CNT))
        #print("List of contour:",CNT)
        rad=[]
        Xlist=[]
        Ylist=[]
        for c in CNT:

                Xlist.append(c[0])
                Ylist.append(c[1])
        # loop over the contours
        for c in cnts1:
                
                # compute the center of the contour, then detect the name of the
                # shape using only the contour
                M = cv2.moments(c)
            
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                shape,approx = sd.detect(c)

                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print(shape)
                
                Approx =np.array(approx).tolist()
                App=[]
                List=[]
                
                X,Y=[],[]
                for i in range(len(Approx)):
                        App.append(Approx[i][0])
                for i in range(len(App)):
                        List.append(CNT.index(App[i]))
        print("The edges are:",App)
        print("the center of the contour is",cX,cY)
        
        centerX=[round(((X-150)/2/radius),2) for X in Xlist]
        centerY=[round(((Y-150)/2/radius),2) for Y in Ylist]
        #print(centerX)
        print()
        #print(centerY)
        K=[]
       
        for x in App:
                i = CNT.index(x)
                xd=(centerX[i]-centerX[i-1])
                xdd=(centerX[i]-centerX[i-1])-(centerX[i-1]-centerX[i-2])
                yd=(centerY[i]-centerY[i-1])
                ydd=(centerY[i]-centerY[i-1])-(centerY[i-1]-centerY[i-2])
                k=(xd*ydd-xdd*yd)/((xd**2)+(yd**2))**1.5
                K.append(k)
        print(K)
        r=[1/x for x in K]
        print(r)
        rd=0
        for x in r:
            rd+=abs(x)    
        print("the roundness is:",round(rd/len(App),2))
        
#call the dim function
dim()

##END##
        
