import cv2
import numpy as np
from matplotlib import pyplot as plt

cv2.ocl.setUseOpenCL(False)

def speed(speed_flag, morpho, slow, slow_count, fast_count, position, no_player):
    thresh = 10
    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
    found = False
    i = 0
    for cnt in cnts:
        if (cv2.contourArea(cnt) < 80) or (cv2.contourArea(cnt) > 4000):
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        if y+h < 150 or \
           w > 100 or \
        h > 200 or \
        w < 30 or \
        h < 60:
            continue    
        found = True
        i+=1
        (xp, yp, wp, hp) = cv2.boundingRect(cnt)
    if found and i < 2:
        speed = np.sqrt((xp+int(wp/2)-position[0])**2 + (yp+int(hp/2)-position[1])**2) / no_player
        if speed < 8:
            slow_count += 1
        else:
            fast_count += 1
        position = (xp+int(wp/2),yp+int(hp/2))
    if not(found):
        no_player +=1
    else:
        no_player = 1
    if slow_count > thresh:
        slow_count = 0
        fast_count = 0
        slow += 1
    if fast_count > thresh/3:
        slow_count = 0
        slow = 0
        fast_count = 0
        speed_flag = 1
    if slow > 8 or no_player > 30:
        slow = 0
        speed_flag = 0
        fast_count = 0
    return speed_flag, slow, slow_count, fast_count, position, no_player
    
##import cv2
##import numpy as np
##from matplotlib import pyplot as plt
##
##cv2.ocl.setUseOpenCL(False)
##hist = 120
##
##
##def avg_speed(posit):
##    speed = []
##    (x1,y1) = posit[0]
##    for i in range(1, hist):
##        (x,y) = posit[i]
##        speed.append(np.sqrt((x-x1)**2 + (y-y1)**2))
##        (x1,y1) = (x,y)
##    return np.max(speed)
##
##video = "D:\Guillaume et Thomas test deux cameras en bas 1.mp4"
##cap = cv2.VideoCapture(video)
##
##fps = int(cap.get(cv2.CAP_PROP_FPS ))
##fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
##out = cv2.VideoWriter("processed.avi", fourcc, np.double(fps), (640,360), True)
##
##ksize = 20
##kernel = np.ones((ksize,ksize),np.uint16)
##fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
##count = 0
##neg_count = 0
##position = [0,0]
##positionR = []
##ret, frame = cap.read()
##frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                   interpolation = cv2.INTER_CUBIC)
##fgmask = fgbg.apply(frame)
##thresh = 10
##slow_count = 0
##fast_count = 0
##slow = 0
##text = "IDLE"
##aLB = ()
##yprev = 0
##hprev = 0
##no_player =0
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
####    cv2.imshow('morpho',morpho)
##    (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
##		cv2.CHAIN_APPROX_SIMPLE)
##    morpho = cv2.cvtColor(morpho,cv2.COLOR_GRAY2BGR)
##    found = False
##    i = 0
##    for cnt in cnts:
##        if (cv2.contourArea(cnt) < 80) or (cv2.contourArea(cnt) > 4000):
##            continue
##        (x, y, w, h) = cv2.boundingRect(cnt)
##        if y+h < 150 or \
##           w > 100 or \
##        h > 200 or \
##        w < 30 or \
##        h < 60:
##            continue    
##        found = True
##        i+=1
##        (xp, yp, wp, hp) = cv2.boundingRect(cnt)
##        cv2.rectangle(frame, (xp, yp),(xp+wp,yp+hp), (255, 0, 0), 2)
####        y_sp = yp+hp-yprev-hprev
####        print(y_sp)
####        if np.abs(y_sp) > 5:
####            print(y_sp)
####        yprev = yp
####        hprev = hp
####        LB = []
####        for A in cnt:
####            (xub,yub) = (A[0,0], A[0,1])
####            if yub > (yp+2*h/3):
####                LB.append([xub,yub])
####        if len(LB) >1:
####            aLB = np.array(LB)
####            (xlb, ylb, wlb, hlb) = cv2.boundingRect(aLB)
####            leftmost = tuple(aLB[aLB[:,0].argmin()])
####            cv2.circle(frame, (xp+int(wp/2), yp+int(h/2)), 3, (255,0,0), -1)
####            cv2.circle(frame, (xlb,ylb+hlb), 3, (255,0,0), -1)
####            cv2.rectangle(frame, (xlb, ylb),(xlb+wlb,ylb+hlb), (255, 0, 0), 2)
##    if found and i < 2:
####        if len(position) > 0 and np.sqrt((xlb-position[0][0])**2 + (position[0][1]-ylb+hlb)**2) < 200:
####        position.insert(0, (xlb,ylb+hlb))
##        cv2.circle(frame, (position[0], position[1]), 3, (0,255,0), -1)
##        speed = np.sqrt((xp+int(wp/2)-position[0])**2 + (yp+int(hp/2)-position[1])**2) / no_player
####        print(no_player, speed)
##        if speed < 6:
##            slow_count += 1
##        else:
##            fast_count += 1
####            position.pop()
##        position = (xp+int(wp/2),yp+int(hp/2))
##    if not(found):
##        no_player +=1
##    else:
##        no_player = 1
##
####    print(slow_count)
##    if slow_count > thresh:
##        slow_count = 0
##        fast_count = 0
##        slow += 1
####        print("slow")
##    if fast_count > thresh/3:
##        slow_count = 0
##        slow = 0
##        fast_count = 0
##        text = "fast"
####        print("fast")
##    if slow > 5 or no_player > 30:
####        print("slow")
##        slow = 0
##        text = "slow"
####    cv2.imshow('morpho',morpho)
##    O = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
##    if len(aLB) >1:
##        cv2.circle(O, leftmost, 3, (255,0,0), -1)
##    tow = np.vstack((frame,O))
####    cv2.imshow('ola', tow)
##    
##    cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
##    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
##                1.0, (255, 255, 255), 4)
##    out.write(frame)
##        
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
##cap.release()
##cv2.destroyAllWindows()
##out.release()
##out = None
##cap = None
