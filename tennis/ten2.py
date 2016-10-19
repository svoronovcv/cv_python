import cv2
import numpy as np

##def sliding_window(image, stepSize, windowSize):
##    for y in range(0, image.shape[0]-windowSize[1], stepSize):
##        for x in range(0, image.shape[1]-windowSize[0], stepSize):
##            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output_4.avi',fourcc, 25.0, (640,720), True)
cap = cv2.VideoCapture('/media/pc/ntfs/downloads/Guillaume vs Thomas test deux cameras 2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=50)
min_area = 200
max_area = 3000
mask = np.zeros((360,640,3), dtype=np.uint8)
mask[:, :,:] = 1
##mask1 = cv2.imread('mask1.jpg', 0)
##kernel = np.ones((3,3),np.uint8)
##mask1 = cv2.dilate(mask1,kernel, iterations=1)
##w = 90
##h = 150
ksize = 10
kernel = np.ones((ksize,ksize),np.uint8)

while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    frame = frame*mask
    fgmask = fgbg.apply(frame)
    fgmask = fgmask * (fgmask >200) #*(mask1 < 150)
##    cv2.imshow('frame',fgmask)
    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
##    cv2.imshow('morpho',morpho)
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if (cv2.contourArea(c) < min_area) or (cv2.contourArea(c) > max_area):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    O = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
    tow = np.vstack((frame,O))
##    cv2.imshow('ola', tow)
    out.write(tow)
##    for (x, y, window) in sliding_window(fgmask, stepSize=10, windowSize=(w, h)):
##        if window.shape[0] != h or window.shape[1] != w:
##            continue
##        if np.sum(window) > 90*150*0.1*255:
##            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
##    cv2.imshow("Security Feed", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
out.release()
out=None
cv2.destroyAllWindows()
