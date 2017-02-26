from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

def hog_proc(fgbg, hog, frame, speed, to, xpp, ypp, xp, yp, slow, no_player, sec, fps):
    to+=1
    if to < fps/3:
        return fgbg,to, speed, xpp, ypp, xp, yp, slow, no_player, 0,0
    else:
        to = 0
    stop_flag = 0
    start_flag = 0
    image = frame[200:,:,:]
    fgmask = fgbg.apply(image)
    fgmask = fgmask * (fgmask >200)
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.1)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = rects #non_max_suppression(rects, probs=None, overlapThresh=0.65)
    j = 0
    mind = 120000000
    xc = image.shape[1]/2
    yc = image.shape[0]
    proplay = (0, 0, 0, 0)
    cv2.circle(image, (int(xpp),int(ypp)), 5, (255,0,0), -1)
    for (xA, yA, xB, yB) in pick:
        if (xB - xA) < 50 or (yB - yA) < 75 or ((xB - xA) < 150 and yB > 3*image.shape[0]/4) or yB < image.shape[0]/6:
            continue
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
        dst = min((xA-xc)**2 + (yB-yc)**2,(xB-xc)**2 + (yB-yc)**2)
        pers = np.sum(fgmask[yA:yB,xA:xB])/255 / (np.sum(fgmask[yA:yB,xA:xB] > -10))
        if dst < mind:
            if dst + 150 < mind and pers > 0.01:
                j+=1
                proplay = (xA, yA, xB, yB)
                xp = (xA+xB)/2
                yp = (yA+yB)/2
                mind = dst
            else:
                if yB > yp:
                    proplay = (xA, yA, xB, yB)
                    xp = (xA+xB)/2
                    yp = (yA+yB)/2
                    mind = dst
    speed = np.sqrt((xp-xpp)**2 + (yp-ypp)**2) / no_player
    if j < 1:
        no_player += 1
        speed = 0
    else:
        no_player = 1
    xpp = xp
    ypp = yp
    if speed > 125:
        speed = 0
    if speed < 30:
        slow+=1
        start_flag = 0
    if speed > 50:
        slow = 0
        start_flag = 1
    if no_player > 3*sec or slow > 3*sec:
        stop_flag = 1
##    cv2.circle(image, (int(xpp),int(ypp)), 5, (0,0,255), -1)
##    cv2.rectangle(image, (0,0), (400, 60), (0,0,0), -1)
##    cv2.putText(image, str(slow), (00, 15), cv2.FONT_HERSHEY_SIMPLEX,
##                0.5, (255, 255, 255), 2)
##    cv2.putText(image, str(speed), (00, 30), cv2.FONT_HERSHEY_SIMPLEX,
##                0.5, (255, 255, 255), 2)
##    cv2.putText(image, str(no_player), (00, 45), cv2.FONT_HERSHEY_SIMPLEX,
##                0.5, (255, 255, 255), 2)
##    cv2.rectangle(image, (proplay[0], proplay[1]), (proplay[2], proplay[3]), (0, 255, 0), 2)
##    cv2.imshow('image', image)
##    cv2.waitKey()
    return fgbg,to, speed, xpp, ypp, xp, yp, slow, no_player, start_flag, stop_flag

