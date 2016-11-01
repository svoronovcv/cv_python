import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100)
minLineLength = 100
maxLineGap =50
lines = cv2.HoughLines(edges,1,np.pi/180,150) #,minLineLength,maxLineGap)
##circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,120,
##                           param1=100,param2=30,minRadius=0,maxRadius=0)
##circles = np.uint16(np.around(circles))
##for c in circles[0,:]:
##    cv2.circle(img, (c[0],c[1]),c[2],(0,255,0),2)
##for line in lines:
##    x1,y1,x2,y2 = line[0]
##    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
for rho,theta in lines[:,0,:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
cv2.imshow("edges",edges)
cv2.imshow("lines",img)
cv2.waitKey()

A = cv2.createLineSegmentDetector(_refine = cv2.LSD_REFINE_ADV)
B = A.detect(gray)
C = A.drawSegments(gray,B[0])
plt.imshow(C)
plt.show()
cv2.destroyAllWindows()
