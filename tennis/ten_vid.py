import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_parts(part):
    (_,cnts,_) = cv2.findContours(part, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('11_.avi', fourcc, 30.0, (1280,720), True)
cap = cv2.VideoCapture('/media/pc/ntfs/downloads/Guillaume et Thomas test deux cameras en bas 1.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=50000)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
min_area = 400
max_area = 20000
min_sn_area = 10
max_sn_area = 400
##kernel = np.ones((3,3),np.uint16)
##mask1 = cv2.dilate(mask1,kernel, iterations=1)
##w = 90
##h = 150
(yc, xc) = (526, 645)
ksize = 40
kernel = np.ones((ksize,ksize),np.uint16)
counter = 0
neg_count = 0
counter_thd = 20

##for i in range(17220):
##    ret, frame = cap.read()
    
while(1):
    ret, frame = cap.read()
    if not(ret):
        break
##    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    cv2.imwrite('f.jpg', frame)
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg2.apply(frame)
    fgmask = fgmask * (fgmask >200)
    fgmask2 = fgmask2 * (fgmask2 >200)
##    cv2.imshow('frame',fgmask)
    di = fgmask-fgmask2;
    di = (di <0)*128 + (di>0)*255
    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
##    morpho = cv2.dilate(fgmask, kernel)
##    cv2.imshow('morpho',morpho)
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    find_cnt = False
    di = cv2.cvtColor(np.uint8(di), cv2.COLOR_GRAY2BGR)
    for c in cnts:
        if (cv2.contourArea(c) < min_area) or (cv2.contourArea(c) > max_area):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (y+h) < 230 or \
        (x-xc)>600 or \
        ((x<xc) and (x-xc+w)<-600) or \
        np.abs(y-yc+h)>350 or \
        (((y+h)<yc) and (y-yc+h)<-300) or \
        w > 200 or \
        h > 300 or \
        w < 30 or \
        h < 80:
            continue
##        if(x-np.floor(0.5*w) > 0):
##            sneakers = find_parts(morpho[y+h:y+np.floor(2*h), x-np.floor(0.5*w):x+np.floor(1.5*w)])
##        if(y-np.floor(0.2*h) > 0):
##            hands = find_parts(morpho[y-np.floor(0.2*h):y, x:x+w])
##        if sneakers != None:
##            for cn in sneakers:
##                if (cv2.contourArea(cn) < min_sn_area) or (cv2.contourArea(cn) > max_sn_area):
##                    continue
##                (xS, yS, wS, hS) = cv2.boundingRect(cn)
##                (xS, yS, wS, hS) = (xS+x-np.floor(0.2*w), yS+y+h, wS, hS)
##                if xS < x:
##                    x = np.uint16(xS)
##                if xS+wS > x+w:
##                    w = np.uint16(xS+wS-x)
##                if yS+hS > y+h and yS+hS-y <300:
##                    h = np.uint16(yS+hS-y)
##                cv2.rectangle(frame, (np.uint16(xS), np.uint16(yS)), (np.uint16(xS + wS), np.uint16(yS + hS)), (255, 0, 0), 2)
##
##        if hands != None:
##            for hn in hands:
##                if (cv2.contourArea(hn) > 1000 and cv2.contourArea(hn) < 150):
##                    continue
##                (xH, yH, wH, hH) = cv2.boundingRect(hn)
##                (xH, yH, wH, hH) = (xH, yH+y-np.floor(0.3*h), wH, hH)
##                if yH < y and h+y-yH < 150:
##                    h = np.uint16(h+y-yH)
##                    y = np.uint16(yH)  
##                cv2.rectangle(frame, (np.uint16(x), np.uint16(yH)), (np.uint16(x + w), np.uint16(yH + hH)), (0, 0, 255), 2)
                    
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    out.write(di)
        
##    cv2.imshow("Security Feed", fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
out.release()
out=None
cv2.destroyAllWindows()
