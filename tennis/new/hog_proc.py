from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

def hog_proc(curr, hog, frame, speed, to, xpp, ypp, xp, yp, slow, no_player, sec):
    to+=1
    if to < 10:
        return to, speed, xpp, ypp, xp, yp, slow, no_player, curr
    else:
        to = 0
    image = frame[200:,:,:]
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.1)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = rects #non_max_suppression(rects, probs=None, overlapThresh=0.65)
    j = 0
    mind = 120000000
    xc = image.shape[1]/2
    yc = image.shape[0]
    for (xA, yA, xB, yB) in pick:
        if (xB - xA) < 50 or (yB - yA) < 75 or ((xB - xA) < 150 and yB > 3*image.shape[0]/4) or yB < image.shape[0]/6:
            continue
        j+=1
        dst = min((xA-xc)**2 + (yB-yc)**2,(xB-xc)**2 + (yB-yc)**2)
        if dst < mind:
            if dst + 150 < mind:
                xp = (xA+xB)/2
                yp = (yA+yB)/2
                mind = dst
            else:
                if yB > yp:
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
    if j < 1 or speed > 200:
        speed = 0
    if speed < 30:
        slow+=1
    else:
        slow = 0
    if no_player > 3*sec or slow > 3*sec:
        curr = 0
    else:
        curr = 1
    return to, speed, xpp, ypp, xp, yp, slow, no_player, curr

