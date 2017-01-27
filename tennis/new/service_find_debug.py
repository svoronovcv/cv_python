import cv2
import numpy as np
import find_start_frame_v2 as fsf
##from matplotlib import pyplot as plt
from court_det import find_serv_point as fsp

def find_service_pos(morpho, xc, yc, position, positionR, count, neg_count):
    serv_flag = 0
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    found = False
    for cnt in cnts:
        if (cv2.contourArea(cnt) < 500) or (cv2.contourArea(cnt) > 4000):
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (y+h) < yc-30 or \
        ((x>xc) and (x-xc)< 100) or \
        ((x<xc) and (x-xc+w)>-100) or \
        w > 150 or \
        h > 200 or \
        w < 30 or \
        h < 70:
            continue    
        found = True

        (xp, yp, wp, hp) = cv2.boundingRect(cnt)
        UB = []
        LB = []
        for A in cnt:
            (xub,yub) = (A[0,0], A[0,1])
            if yub < (yp+h/2):
                UB.append([xub,yub])
            else:
                LB.append([xub,yub])
        
        if len(UB)>4 and len(LB)>4:
            aUB = np.array(UB)
            (xub, yub, wub, hub) = cv2.boundingRect(aUB)
            aLB = np.array(LB)
            (xlb, ylb, wlb, hlb) = cv2.boundingRect(aLB)
            ellipse_ub = cv2.fitEllipse(aUB)
            _,_,D = ellipse_ub
            ellipse_lb = cv2.fitEllipse(aLB)
            _,_,DL = ellipse_lb
            if x>xc and D >100 and D <175 and DL > 10 and DL < 65:
                count+=1
                neg_count = 0
            elif x<xc and D >5 and D <60 and DL > 100 and DL < 160:
                count+=1
                neg_count = 0
            
    ##        ellipse = cv2.fitEllipse(cnt)
    ##        cv2.ellipse(morpho,ellipse,(0,255,0),2)
    ##        A,B,C = ellipse
    ##        if B[1]/B[0] < 3.5:
    ##            count+=1
    ##            neg_count = 0
            else:
                neg_count += 1
                serv_flag = 0
            if neg_count > 10:
                position = []
                positionR = []
                count = 0
            if count > 25 and (fsf.avg_speed(position) < 1 or fsf.avg_speed(positionR) < 1):
                serv_flag = 1
                count = 0
    if found:
        position.insert(0, (xp,yp+hp))
        positionR.insert(0, (xp+wp,yp+hp))
    return serv_flag, position, positionR, count, neg_count

##cv2.ocl.setUseOpenCL(False)
##video = "/media/pc/ntfs/downloads/Guillaume et Thomas test deux cameras en haut 1.mp4"
##cap = cv2.VideoCapture(video)
##ksize = 10
##kernel = np.ones((ksize,ksize),np.uint16)
##fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
##count = 0
##neg_count = 0
##position = []
##positionR = []
##ret, frame = cap.read()
##frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                   interpolation = cv2.INTER_CUBIC)
##fgmask = fgbg.apply(frame)
##xc, yc = fsp(frame)
##while(1):
##    ret, frame = cap.read()
##    if not(ret):
##        break
##    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
##    fgmask = fgbg.apply(frame)
##    fgmask = fgmask * (fgmask >200)
##    morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
##    
####    cv2.imwrite("mm.jpg", morpho)
##    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
##		cv2.CHAIN_APPROX_SIMPLE)
##    morpho = cv2.cvtColor(morpho,cv2.COLOR_GRAY2BGR)
##    found = False
##
##    for cnt in cnts:
##        if (cv2.contourArea(cnt) < 500) or (cv2.contourArea(cnt) > 4000):
##            continue
##        (x, y, w, h) = cv2.boundingRect(cnt)
##        if (y+h) < yc-30 or \
##        ((x>xc) and (x-xc)< 100) or \
##        ((x<xc) and (x-xc+w)>-100) or \
##        w > 150 or \
##        h > 200 or \
##        w < 30 or \
##        h < 70:
##            continue    
##        found = True
##
##        (xp, yp, wp, hp) = cv2.boundingRect(cnt)
##        UB = []
##        LB = []
##        for A in cnt:
##            (xub,yub) = (A[0,0], A[0,1])
##            if yub < (yp+h/2):
##                UB.append([xub,yub])
##            else:
##                LB.append([xub,yub])
##        
##        if len(UB)>4 and len(LB)>4:
##            aUB = np.array(UB)
##            (xub, yub, wub, hub) = cv2.boundingRect(aUB)
##            aLB = np.array(LB)
##            (xlb, ylb, wlb, hlb) = cv2.boundingRect(aLB)
##            ellipse_ub = cv2.fitEllipse(aUB)
##            cv2.ellipse(morpho,ellipse_ub,(255,0,0),2)
##            _,_,D = ellipse_ub
##            ellipse_lb = cv2.fitEllipse(aLB)
##            cv2.ellipse(morpho,ellipse_lb,(255,0,0),2)
##            _,_,DL = ellipse_lb
##            if x>xc and D >100 and D <175 and DL > 10 and DL < 65:
##                count+=1
##                neg_count = 0
##            elif x<xc and D >5 and D <60 and DL > 100 and DL < 160:
##                count+=1
##                neg_count = 0
##            
##    ##        ellipse = cv2.fitEllipse(cnt)
##    ##        cv2.ellipse(morpho,ellipse,(0,255,0),2)
##    ##        A,B,C = ellipse
##    ##        if B[1]/B[0] < 3.5:
##    ##            count+=1
##    ##            neg_count = 0
##            else:
##                neg_count += 1
##            if count > 10:
##                print(count)
##            if neg_count > 10:
##                position = []
##                positionR = []
##                count = 0
##            if count > 25 and (fsf.avg_speed(position) < 1 or fsf.avg_speed(positionR) < 1):
##    ##            print(count, C, B[1]/B[0], fsf.avg_speed(position), fsf.avg_speed(positionR))
##                plt.imshow(frame)
##                plt.show()
##                count = -100
##    if found:
##        position.insert(0, (xp,yp+hp))
##        positionR.insert(0, (xp+wp,yp+hp))
##
##    cv2.imshow('morpho',morpho)
##    O = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
##    tow = np.vstack((frame,O))
##    cv2.imshow('ola', tow)
##    k = cv2.waitKey(30) & 0xff
##    if k == 27:
##        break
##    if k == ord('q'):
##        plt.imshow(frame)
##        plt.show()
##    if k == ord('a'):
##        for i in range(30):
##            ret, frame = cap.read()
##            frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
##            fgmask = fgbg.apply(frame)
##            if not(ret):
##                break
##    if k == ord('s'):
##        for i in range(30*10):
##            ret, frame = cap.read()
##            frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
##            fgmask = fgbg.apply(frame)
##            if not(ret):
##                break
##    if k == ord('d'):
##        for i in range(30*30):
##            ret, frame = cap.read()
##            frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
##            fgmask = fgbg.apply(frame)
##            if not(ret):
##                break
##        
##cap.release()
##cv2.destroyAllWindows()
