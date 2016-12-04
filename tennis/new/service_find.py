import cv2
import numpy as np
import find_start_frame_v2 as fsf
from matplotlib import pyplot as plt

cv2.ocl.setUseOpenCL(False)
video = "D:\Guillaume et Thomas test deux cameras en bas 1.mp4"
cap = cv2.VideoCapture(video)
ksize = 10
kernel = np.ones((ksize,ksize),np.uint16)
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
count = 0
neg_count = 0
position = []
positionR = []
xc = 320
while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
    fgmask = fgbg.apply(frame)
    fgmask = fgmask * (fgmask >200)
    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
##    cv2.imwrite("mm.jpg", morpho)
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    morpho = cv2.cvtColor(morpho,cv2.COLOR_GRAY2BGR)
    found = False
    for cnt in cnts:
        if (cv2.contourArea(cnt) < 500) or (cv2.contourArea(cnt) > 4000):
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (y+h) < 250 or \
        ((x>xc) and (x-xc)< 100) or \
        ((x<xc) and (x-xc+w)>-100) or \
        w > 150 or \
        h > 200 or \
        w < 30 or \
        h < 70:
            continue    
        found = True
        (xp, yp, wp, hp) = cv2.boundingRect(cnt)
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(morpho,ellipse,(0,255,0),2)
        A,B,C = ellipse
        if B[1]/B[0] < 3.5:
            count+=1
            neg_count = 0
        else:
            neg_count += 1
        if neg_count > 5:
            position = []
            positionR = []
            count = 0
        if count > 50 and (fsf.avg_speed(position) < 1 or fsf.avg_speed(positionR) < 1):
            print(count, C, B[1]/B[0], fsf.avg_speed(position), fsf.avg_speed(positionR))
            plt.imshow(frame)
            plt.show()
            count = -20
    if found:
        position.insert(0, (xp,yp+hp))
        positionR.insert(0, (xp+wp,yp+hp))

    cv2.imshow('morpho',morpho)
    O = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
    tow = np.vstack((frame,O))
    cv2.imshow('ola', tow)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord('q'):
        plt.imshow(frame)
        plt.show()
    if k == ord('a'):
        for i in range(30):
            ret, frame = cap.read()
            fgbg.apply(frame)
            if not(ret):
                break
    if k == ord('s'):
        for i in range(30*10):
            ret, frame = cap.read()
            fgbg.apply(frame)
            if not(ret):
                break
    if k == ord('d'):
        for i in range(30*30):
            ret, frame = cap.read()
            fgbg.apply(frame)
            if not(ret):
                break
        
cap.release()
cv2.destroyAllWindows()
