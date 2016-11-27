import cv2
import numpy as np
##import matplotlib.pyplot as plt

def projX(frame,x,y,w,h):
    return np.sum(frame[y:y+h,x:x+w],0)/255/h

def projY(frame,x,y,w,h):
    return np.sum(frame[y:y+h,x:x+w],1)/255/w

def find_parts(part):
    (_,cnts,_) = cv2.findContours(part, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    return cnts
def avg_speed(posit):
    hist = 25
    speed = []
    (x1,y1) = posit[0]
    for i in range(1, hist):
        (x,y) = posit[i]
        speed.append(np.sqrt((x-x1)**2 + (y-y1)**2))
        (x1,y1) = (x,y)
    return np.median(speed)

def find_player(outflag, morpho, cnts, frame, min_area, max_area, min_sn_area, max_sn_area, xc, yc):
    find_pl = False
    (x,y,w,h) = (0,0,0,0)
    (xA,yA,wA,hA) = (0,0,0,0)
    for c in cnts:
        if (cv2.contourArea(c) < min_area) or (cv2.contourArea(c) > max_area):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (y+h) < 150 or \
        (x-xc)>100 or \
        ((x<xc) and (x-xc+w)<-100) or \
        np.abs(y-yc+h)>60 or \
        (((y+h)<yc) and (y-yc+h)<-20) or \
        w > 100 or \
        h > 150 or \
        w < 30 or \
        h < 80:
            continue
        if ((x-xc)**2 + (y-yc)**2) < ((x-xA)**2 + (y-yA)**2):
            (xA,yA,wA,hA) = (x,y,w,h)
            find_pl = True
    if find_pl:
        (x,y,w,h) = (xA,yA,wA,hA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##        cv2.imshow("pla", frame)
##        cv2.waitKey(30)
        if (outflag > 0):
            sneakers = find_parts(morpho[y+h:y+np.floor(2*h), \
                                         x-np.floor(0.5*w):x+np.floor(1.5*w)])
            if sneakers != None:
                for cn in sneakers:
                    if (cv2.contourArea(cn) < min_sn_area) or \
                    (cv2.contourArea(cn) > max_sn_area):
                        continue
                    (xS, yS, wS, hS) = cv2.boundingRect(cn)
                    (xS, yS, wS, hS) = (xS+x-np.floor(0.5*w), yS+y+h, wS, hS)
                    if xS < x:
                        x = np.uint16(xS)
                    if xS+wS > x+w:
                        w = np.uint16(xS+wS-x)
                    if yS+hS > y+h and yS+hS-y <150:
                        h = np.uint16(yS+hS-y)
                    cv2.rectangle(frame, (np.uint16(xS), np.uint16(yS)), \
                                  (np.uint16(xS + wS), np.uint16(yS + hS)), (255, 0, 0), 2)

            hands = find_parts(morpho[y-np.floor(0.2*h):y, x:x+w])
            if hands != None:
                for hn in hands:
                    if (cv2.contourArea(hn) > 500 and cv2.contourArea(hn) < 150):
                        continue
                    (xH, yH, wH, hH) = cv2.boundingRect(hn)
                    (xH, yH, wH, hH) = (xH, yH+y-np.floor(0.2*h), wH, hH)
                    if yH < y and h+y-yH < 150:
                        h = np.uint16(h+y-yH)
                        y = np.uint16(yH)  
                    cv2.rectangle(frame, (np.uint16(x), np.uint16(yH)), \
                                  (np.uint16(x + w), np.uint16(yH + hH)), (0, 0, 255), 2)   
    return find_pl, x, y, w, h

def find_start(fgmask,x,y,w,h,count,pos, posR):
    found = 0
    if avg_speed(pos) < 3 or avg_speed(posR) < 3:
        A = projX(fgmask,x,y,w,h)
        B = projY(fgmask,x,y,w,h)
        if np.max(B[:np.uint16(h*0.2)]) > 0 and \
           np.max(B[np.uint16(h*0.2):]) / np.max(B[:np.uint16(h*0.2)]) > 3 and \
            np.std(B[np.uint16(h*0.2):]) < 0.2 and \
            h/w > 2 and \
            np.mean(B[np.uint16(h*0.4):np.uint16(h*0.8)]) \
            / np.mean(B[:np.uint16(h*0.2)]) > 2:
            count = -20
            found = 1
##            plt.figure()
##            plt.plot(A)
##            plt.subplot(211)
##            plt.plot(A)
##            plt.subplot(212)
##            plt.plot(B)
##            plt.show()
    return count, found
    
def find(outflag, frame, fgbg,  min_area, max_area, min_sn_area, max_sn_area, \
             mask, mask1, yc, xc, kernel, counter, neg_count, \
             counter_thd, position, positionR):
    flag = 0
    loc_frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
##    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    loc_frame = loc_frame*(mask1 > 180)
##    cv2.imwrite('f.jpg', frame)
    fgmask = fgbg.apply(loc_frame)
    fgmask = fgmask * (fgmask >200) *(mask1[:,:,0] > 200)
##    cv2.imshow('frame',fgmask)
##    cv2.waitKey(1)
    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
##    morpho = cv2.dilate(fgmask, kernel)
##    cv2.imshow('morpho',morpho)
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    find_cnt, x ,y, w, h = find_player(outflag, morpho, cnts,loc_frame, min_area, max_area, min_sn_area, max_sn_area, xc, yc)
    if find_cnt:
        if counter > counter_thd:
            counter, flag = find_start(fgmask,x,y,w,h,counter, position, positionR)
        counter += 1
        neg_count = 0
        position.insert(0, (x,y+h))
        positionR.insert(0, (x+w,y+h))
    else:
        neg_count += 1
        if neg_count > 20:
            counter = 0
            position = []
            positionR = []
    return fgbg, counter, neg_count, position, positionR, flag
