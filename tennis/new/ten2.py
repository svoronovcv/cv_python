import cv2
import numpy as np
import matplotlib.pyplot as plt

##def sliding_window(image, stepSize, windowSize):
##    for y in range(0, image.shape[0]-windowSize[1], stepSize):
##        for x in range(0, image.shape[1]-windowSize[0], stepSize):
##            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def projX(frame,x,y,w,h):
    return np.sum(frame[y:y+h,x:x+w],0)/255/h
def projY(frame,x,y,w,h):
    return np.sum(frame[y:y+h,x:x+w],1)/255/w
def find_parts(part):
    (_,cnts,_) = cv2.findContours(part, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    return cnts
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
##out = cv2.VideoWriter('output_5.avi', fourcc, 30.0, (640,720), True)
cap = cv2.VideoCapture("D:\Guillaume et Thomas test deux cameras en bas 1.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
min_area = 80
max_area = 4000
min_sn_area = 10
max_sn_area = 200
mask = np.zeros((360,640,3), dtype=np.uint16)
mask[100:, :,:] = 1
mask1 = cv2.imread('ff.jpg')
##kernel = np.ones((3,3),np.uint16)
##mask1 = cv2.dilate(mask1,kernel, iterations=1)
##w = 90
##h = 150
(yc, xc) = (263, 323)
ksize = 10
kernel = np.ones((ksize,ksize),np.uint16)
counter = 0
neg_count = 0
counter_thd = 20

##for i in range(17250):
##    ret, frame = cap.read()
    
##frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
##    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    frame = frame*(mask1 > 180)
    cv2.imwrite('f.jpg', frame)
    fgmask = fgbg.apply(frame)
    fgmask = fgmask * (fgmask >200) *(mask1[:,:,0] > 200)
##    cv2.imshow('frame',fgmask)
    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
##    morpho = cv2.dilate(fgmask, kernel)
    cv2.imshow('morpho',morpho)
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    find_cnt = False
    for c in cnts:
        if (cv2.contourArea(c) < min_area) or (cv2.contourArea(c) > max_area):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (y+h) < 150 or \
        (x-xc)>100 or \
        ((x<xc) and (x-xc+w)<-100) or \
        np.abs(y-yc+h)>40 or \
        (((y+h)<yc) and (y-yc+h)<-50) or \
        w > 150 or \
        h > 200 or \
        w < 30 or \
        h < 70:
            continue
##        sneakers = find_parts(morpho[y+h:y+np.floor(2*h), x-np.floor(0.5*w):x+np.floor(1.5*w)])
##        hands = find_parts(morpho[y-np.floor(0.2*h):y, x:x+w])
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
##                if yS+hS > y+h and yS+hS-y <150:
##                    h = np.uint16(yS+hS-y)
##                cv2.rectangle(frame, (np.uint16(xS), np.uint16(yS)), (np.uint16(xS + wS), np.uint16(yS + hS)), (255, 0, 0), 2)
##
##        if hands != None:
##            for hn in hands:
##                print(cv2.contourArea(hn))
##                if (cv2.contourArea(hn) > 500 and cv2.contourArea(hn) < 150):
##                    continue
##                (xH, yH, wH, hH) = cv2.boundingRect(hn)
##                (xH, yH, wH, hH) = (xH, yH+y-np.floor(0.3*h), wH, hH)
##                if yH < y and h+y-yH < 150:
##                    h = np.uint16(h+y-yH)
##                    y = np.uint16(yH)  
##                cv2.rectangle(frame, (np.uint16(x), np.uint16(yH)), (np.uint16(x + w), np.uint16(yH + hH)), (0, 0, 255), 2)
                    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        find_cnt = True
        if counter > counter_thd:
            A = projX(fgmask,x,y,w,h)
            B = projY(fgmask,x,y,w,h)
            k = cv2.waitKey(50) & 0xff
            if k == ord('q'):
                O = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
                tow = np.vstack((frame,O))
                cv2.imshow('ola', tow)
                plt.figure()
                plt.plot(A)
                plt.subplot(211)
                plt.plot(A)
                plt.subplot(212)
                plt.plot(B)
                plt.show()
                print(np.max(B[np.uint16(h*0.2):]) / np.max(B[:np.uint16(h*0.2)]),np.std(B[np.uint16(h*0.2):]),h/w,np.mean(B[np.uint16(h*0.4):np.uint16(h*0.8)]) / np.mean(B[:np.uint16(h*0.2)]))
            if np.max(B[np.uint16(h*0.2):]) / np.max(B[:np.uint16(h*0.2)]) > 3 and \
               np.std(B[np.uint16(h*0.2):]) < 0.2 and \
               h/w > 2 and \
               np.mean(B[np.uint16(h*0.4):np.uint16(h*0.8)]) / np.mean(B[:np.uint16(h*0.2)]) > 2:
                counter = -20
                plt.figure()
                plt.plot(A)
                plt.subplot(211)
                plt.plot(A)
                plt.subplot(212)
                plt.plot(B)
                plt.show()
    if find_cnt:
        counter += 1
        neg_count = 0
        print(counter)
    else:
        neg_count += 1
        if neg_count > 20:
            counter = 0
        
##    out.write(tow)
##    for (x, y, window) in sliding_window(fgmask,stepSize=10,windowSize=(w, h)):
##        if window.shape[0] != h or window.shape[1] != w:
##            continue
##        if np.sum(window) > 90*150*0.1*255:
##            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
##    cv2.imshow("Security Feed", frame)
    O = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
    tow = np.vstack((frame,O))
    cv2.imshow('ola', tow)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
##out.release()
##out=None
cv2.destroyAllWindows()
