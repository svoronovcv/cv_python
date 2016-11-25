import cv2
import numpy as np
import matplotlib.pyplot as plt

#Functions for start frame detection--------------------------------------------------
def projX(frame,x,y,w,h):
    return np.sum(frame[y:y+h,x:x+w],0)/255/h
def projY(frame,x,y,w,h):
    return np.sum(frame[y:y+h,x:x+w],1)/255/w
# Function for finding sneakers or hands
def find_parts(part):
    (_,cnts,_) = cv2.findContours(part, cv2.RETR_CCOMP,
		cv2.CHAIN_APPROX_SIMPLE)
    return cnts
#---------------------------------------------------------------------------------------

# Start frame id, if it is a start frame - add 1 to D and return, otherwise
# add 0 to D and keep searching--------------------------------------------
def find(cap,fgbg,D):
    # Parameters' initialization--------------------------------------------------------
    min_area = 80         # min area for player detection
    max_area = 4000       # max area for player detection
    min_sn_area = 10      # min area for sneakers detection
    max_sn_area = 200     # max area for sneakers detection
    mask = np.zeros((360,640,3), dtype=np.uint16) # mask for court area
    mask[100:, :,:] = 1
    (yc, xc) = (217, 307) # serive point
##    (yc, xc) = (230, 345) # serive point
    ksize = 20            # kernel size for morphological operations
    kernel = np.ones((ksize,ksize),np.uint16) # kernel for morphological operations
    counter = 0           # counter for service id
    neg_count = 0         # counter for being not in sirvice area
    counter_thd = 20      # threshold for service id counter
    #-----------------------------------------------------------------------------------
    
    while(1):
        ret, frame = cap.read()
        if not(ret):
            return ret, D
        frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                           interpolation = cv2.INTER_CUBIC) # resize frames for a speed-up
        frame = frame #*(mask1 > 180)
##        cv2.imwrite('f.jpg', frame)
        fgmask = fgbg.apply(frame) # apply foreground segmentation
        fgmask = fgmask * (fgmask >200) #*(mask1[:,:,0] > 200)
        morpho = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) # apply moprhology to strengthen player area
##        cv2.imshow('morpho',morpho)
        (_,cnts,_) = cv2.findContours(morpho, cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE) # find counters for player search
        find_cnt = False # Flag for player in service area
        
        # Check counters if there is a player-----------------------------------------------------
        for c in cnts:
            if (cv2.contourArea(c) < min_area) or (cv2.contourArea(c) > max_area):
                continue # if there it is not a player in service area - skip
            (x, y, w, h) = cv2.boundingRect(c) # Bounding rect for a possible player

            # Requirements for a counter to be a player--------------------------------------------
            if (y+h) < 150 or \
            (x-xc)>100 or \
            ((x<xc) and (x-xc+w)<-100) or \
            np.abs(y-yc+h)>40 or \
            (((y+h)<yc) and (y-yc+h)<-30) or \
            w > 150 or \
            h > 150 or \
            w < 30 or \
            h < 60:
                continue
            sneakers = find_parts(morpho[y+h:y+np.floor(2*h), x-np.floor(0.5*w):x+np.floor(1.5*w)]) # find sneakers
            hands = find_parts(morpho[y-np.floor(0.2*h):y, x:x+w]) # Find hands
            if sneakers != None:
                # Add sneakers and modify the bounding rect-----------------------------------------
                for cn in sneakers:
                    if (cv2.contourArea(cn) < min_sn_area) or (cv2.contourArea(cn) > max_sn_area):
                        continue
                    (xS, yS, wS, hS) = cv2.boundingRect(cn)
                    (xS, yS, wS, hS) = (xS+x-np.floor(0.2*w), yS+y+h, wS, hS)
                    if xS < x:
                        x = np.uint16(xS)
                    if xS+wS > x+w:
                        w = np.uint16(xS+wS-x)
                    if yS+hS > y+h and yS+hS-y <150:
                        h = np.uint16(yS+hS-y)
                    cv2.rectangle(frame, (np.uint16(xS), np.uint16(yS)), (np.uint16(xS + wS), np.uint16(yS + hS)), (255, 0, 0), 2)

            if hands != None:
                # Find hands and modify the bounding rect--------------------------------------------
                for hn in hands:
                    if (cv2.contourArea(hn) > 500 and cv2.contourArea(hn) < 150):
                        continue
                    (xH, yH, wH, hH) = cv2.boundingRect(hn)
                    (xH, yH, wH, hH) = (xH, yH+y-np.floor(0.3*h), wH, hH)
                    if yH < y and h+y-yH < 150:
                        h = np.uint16(h+y-yH)
                        y = np.uint16(yH)  
                    cv2.rectangle(frame, (np.uint16(x), np.uint16(yH)), (np.uint16(x + w), np.uint16(yH + hH)), (0, 0, 255), 2)
                        
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            find_cnt = True
            
            # If player in the service area for more then a threshold frames - check if it is a start frame--------------------
            if counter > counter_thd:
                A = projX(fgmask,x,y,w,h)
                B = projY(fgmask,x,y,w,h)
                k = cv2.waitKey(50) & 0xff
                
                # Check the start frame requirements---------------------------------------------------
                if np.max(B[np.uint16(h*0.2):]) / np.max(B[:np.uint16(h*0.2)]) > 3 and \
                   np.std(B[np.uint16(h*0.2):]) < 0.2 and \
                   h/w > 2 and \
                   np.mean(B[np.uint16(h*0.4):np.uint16(h*0.8)]) / np.mean(B[:np.uint16(h*0.2)]) > 2:
                    D.append(1) # Add start frame flag to the flags array
                    return ret, D
        if find_cnt: # If the player in the service area - increase the counter and reset neg counter
            counter += 1
            neg_count = 0
        else:
            neg_count += 1
            if neg_count > 20: # If there is no player in the service area - reset the in- counter
                counter = 0
        D.append(0) # Add IDL flag to the flags array
