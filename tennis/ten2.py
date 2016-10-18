import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('Guillaume vs Thomas test deux cameras 2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=50)
min_area = 600
max_area = 5000
mask = np.zeros((360,640,3), dtype=np.uint8)
mask[200:, :,:] = 1

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    frame = frame*mask
    fgmask = fgbg.apply(frame)
    fgmask = fgmask * (fgmask >200)
    cv2.imshow('frame',fgmask)
    kernel = np.ones((10,10),np.uint8)
    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('morpho',morpho)
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if (cv2.contourArea(c) < min_area) or (cv2.contourArea(c) > max_area):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("Security Feed", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
