import numpy as np
import cv2



## Function get VideoCapture and fps and return VideoCapture at the moment of ending of the frame
## Change D value to 2 if it is not end-frame, 3 for end-frame point
def fnd_endframe(cap,fps,D):
##init some flags and const for algorithm----------------------------    
    min_ball_area = 9
    max_ball_area = 50
    min_dif_border = 10
    max_dif_border = 190
    max_sec = 4.0
    greenLower = (15, 100, 50)
    greenUpper = (52, 150, 255)
    canny_thr = 100
    border = max_sec*fps
    counter = 0
##--------------------------------------------------------------------
    
##Preparation for processing---------------------------------------------    
    ret,init_frame =  cap.read()#read first frame
    if ret:
        (height,width,depth) = (init_frame.shape)
    D.append(2)#append flag 'GAME'
    
    if not (ret):
        return cap,D
    while (True):
        ret,frame = cap.read()
        if not (ret):
            return ret,D
        if (counter < border):
            D.append(2)#if we find ball some times ago
        else:
            D.append(3)#if we dont find ball too much times
            return ret,D
##--------------------------------------------------------------------------
            
##ball search processing------------------------------------------------------------------------------------------------------------
##Image and mask preparation-----------------------------------------------------------------------------------------------------
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(frame_hsv, greenLower, greenUpper)##weak checking via color
        #yellow = cv2.erode(yellow, None, iterations=2)
        yellow = cv2.dilate(yellow, None, iterations=2)
        
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame,(5,5),0)
        
        frame_p = init_frame
        gray_frame_p= cv2.cvtColor(frame_p,cv2.COLOR_BGR2GRAY)
        gray_frame_p = cv2.GaussianBlur(gray_frame_p,(5,5),0)
        
        dif = abs(gray_frame-gray_frame_p)#find differnce between two back-to-back frames to detect moving objects
        mask = cv2.inRange(dif,min_dif_border,max_dif_border)
        
        edges = cv2.Canny(gray_frame, canny_thr, 2*canny_thr)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        (a,cnts,b) = cv2.findContours(closed, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)##find all contours in image
##--End preparation-------------------------------------------------------------------------------------------------------------
        
 ##Contours processing---------------------------------------------------------------------------------------------------------
 ##We find all contours admissible size, shape and color plus it shuold be
        c_number = 0#number of good contours        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            (x,y),r = cv2.minEnclosingCircle(c)
            circle_sq = np.pi * (r**2)
            area = cv2.contourArea(c)
            ratio = area/circle_sq
            r = int(r)
            y = int(y)
            x = int(x)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            a1 = box[0]-box[1]
            a2 = box[1]-box[2]
            side_ratio = np.linalg.norm(a1)/np.linalg.norm(a2)
            box = np.int0(box)
    
            if (x > 0) and (x < width) and (y > 0) and (y< height) and (np.pi *ratio< 3.3) and (mask[y,x] > 0) and (area < max_ball_area) \
            and (area > min_ball_area) and (len(approx) < 15) and yellow[y,x] > 0  and side_ratio > 0.8 and side_ratio < 1.25:
                c_number += 1
##---End contour processing-------------------------------------------------------------------------------------------------
##Switch some flags and counters-for future processing----------------------------------------------------------------------                
        if c_number > 0 and c_number < 3:
            counter = 0
        else:
            counter += 1
            
        init_frame = frame ##move previous frame valu to current frame
##-------------------------------------------------------------------------------------------------------------------------------------            
