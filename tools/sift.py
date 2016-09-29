import cv2
import sys
import numpy as np

img = cv2.imread('./img/1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color= (51,163,236))

cv2.imshow('sift points', img)
cv2.waitKey()
cv2.destroyAllWindows()
