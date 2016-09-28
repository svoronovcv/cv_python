import cv2
import numpy as np

img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
minLineLength = 50
maxLineGap =10
lines = cv2.HoughLinesP(edges,1,np.pi/180*45,100,minLineLength,maxLineGap)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,120,
                           param1=100,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for c in circles[0,:]:
    cv2.circle(img, (c[0],c[1]),c[2],(0,255,0),2)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("edges",edges)
cv2.imshow("lines",img)
cv2.waitKey()
cv2.destroyAllWindows()
