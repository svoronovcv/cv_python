import cv2
import numpy as np

img = cv2.imread('1.jpg', 0)
cv2.imshow('canny', cv2.Canny(img, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()

ret, thresh = cv2.threshold(img, 127,255,0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0),2)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()

img = cv2.imread('1.jpg', 0)
ret, thresh = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),3)
    (x,y),radius = cv2.minEnclosingCircle(c)
    center = (int(x),int(y))
    radius = int(radius)
    img = cv2.circle(img, center,radius,(0,255,0),2)
    epsilon = 0.01*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    hull = cv2.convexHull(c)
    cv2.drawContours(img,[hull],0,(128,128,128),5)
cv2.drawContours(img,contours,-1,(255,0,0),1)
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()
