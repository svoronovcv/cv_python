import cv2
import numpy as np
import matplotlib.pyplot as plt

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
##out = cv2.VideoWriter('output_5.avi', fourcc, 25.0, (640,720), True)
cap = cv2.VideoCapture('/media/pc/ntfs/downloads/Waterloo tennis Rodrigo 15_06_2016.mp4')

for i in range(6000):
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)

r,h,c,w = 150,120,260,90
track_window = (c,r,w,h)
roi = frame[r:r+h, c:c+w]
hsv_roi =  roi #cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
##mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([roi],[0,1], None,[256,256],[0,256,0,256])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
    hsv = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,256,0,256],1)
 
    # apply meanshift to get the new location
    rett, track_window = cv2.meanShift(dst, track_window, term_crit)
 
    # Draw it on image
##    pts = cv2.boxPoints(rett)
##    pts = np.int0(pts)
##    img2 = cv2.polylines(frame,[pts],True, 255,2)
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv2.imshow('img2',img2)

##    out.write(tow)      
    cv2.imshow("Security Feed", img2)
    print(track_window)
    cv2.imwrite('f.jpg', img2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
##out.release()
##out=None
cv2.destroyAllWindows()
