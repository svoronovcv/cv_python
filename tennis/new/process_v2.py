import cv2
import numpy as np
import os
import sys
from argparse import ArgumentParser
from sys import argv
from timeit import default_timer as timer
import find_start_frame_v2 as firstf
from ball_tracking_v2 import fnd_endframe as endf
from court_det import find_serv_point as f_sp
from service_find import find_service_pos as f_s_p
from player_speed import speed

cv2.ocl.setUseOpenCL(False)
def process(fname, fout, outdoor):
    start = timer()
    D = []

    # Init video in and out video
    video = fname
    cap = cv2.VideoCapture(video)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS ))
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    out = cv2.VideoWriter(fout+".avi", fourcc, np.double(fps), (Width,Height), True)
    if out.isOpened():
        print("Video file has been created!") # Check if file is sucessfully created
    else:
        print("Can not create the output file")
    # Create foreground extracter
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
text_to_put = {
    0: 'IDLE',
    1: 'SERVICE DETECTED',
    2: 'GAME',
    3: 'LOST BALL',
    4: 'SERVICE POSITION',
    5: 'HIGH SPEED',
    6: 'SLOW PLAYER'
} # Dictionary for flags

## Start frame flag search parameters-------------------------------
min_area = 80
max_area = 4000
min_sn_area = 10
max_sn_area = 200
mask = np.zeros((int(Height/2), int(Width/2),3), dtype=np.uint16)
mask[100:, :,:] = 1
mask1 = np.zeros((int(Height/2), int(Width/2),3), dtype=np.uint16)+255
ksize = 15 + 5*args.outdoor
kernel = np.ones((ksize,ksize),np.uint16)
counter = 0
neg_count = 0
counter_thd = 25
position = []
positionR = []
serv_neg_count = 0
serv_counter = 0
serv_position = []
serv_positionR = []
##------------------------------------------------------------------

## End frame flag search parameters-------------------------------
min_ball_area = 9
max_ball_area = 50
min_dif_border = 10
max_dif_border = 190
max_sec = 3.5
greenLower = (15, 100, 50)
greenUpper = (52, 150, 255)
canny_thr = 100
border = max_sec*fps
end_counter = 0
found_max_sec = 2
nf_max_sec = 1
thresh_ball_fond = found_max_sec*fps
thresh_ball_nf = nf_max_sec*fps
ball_cont_found = 0
##------------------------------------------------------------------

## HoG search parameters-------------------------------
winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = True
nlevels = 16
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog_to = 0
hog_speed = 0
hog_xp , hog_yp = 0, 0
hog_xpp , hog_ypp = 0, 0
hog_no_player = 1
hog_slow=0
hog_sec = 10
hog_flag = 0
##------------------------------------------------------------------

## Do the first iteration in order to obtain init_frame for end_frame search
ret, frame = cap.read()
##loc_frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
(xc,yc) = (320,200) #f_sp(loc_frame)
##cv2.imwrite('n2.jpg',loc_frame)
##print(xc,yc)

##cv2.imwrite('dfg.jpg', loc_frame)
morpho, fgbg, counter, neg_count, position, positionR, start_flag = firstf.find(args.outdoor, frame, fgbg, \
                                                                  min_area, \
                                                                  max_area, \
                                                                  min_sn_area,\
                                                                  max_sn_area, \
                                                                  mask, mask1, \
                                                                  yc, xc, \
                                                                  kernel, \
                                                                  counter, \
                                                                  neg_count, \
                                                                  counter_thd, \
                                                                  position, \
                                                                  positionR) # find a service frame
serv_flag, serv_position, serv_positionR, serv_counter, serv_neg_count = f_s_p(morpho, \
                                                                               xc, yc, \
                                                                               serv_position, \
                                                                               serv_positionR, \
                                                                               serv_counter, \
                                                                               serv_neg_count)
init_frame = frame
D.append(0)
curr = 0
f =0
nf_counter = 0
# Process the video
it = 0
speed_flag = 0
print("The video file is being processed:")
while(1):
    
########    READ FRAME
    ret, frame = cap.read()
    it += 1
    if not(ret): # stop if it the end of the video
        break
    
########    UPDATE OUTPUT
    
    procent = np.round(1000*it/frames_num)/10
    if procent > 100:
        procent = 100
    sys.stdout.write('\r%s %%' % procent)
    sys.stdout.flush()
    
########    HoG SPEED
    
    hog_to, hog_speed, hog_xpp, hog_ypp, hog_xp, hog_yp, hog_slow, hog_no_player, hog_flag = hog_proc(hog_flag, hog, frame, hog_speed, \
                                                                                          hog_to, hog_xpp, \
                                                                                          hog_ypp, hog_xp, \
                                                                                          hog_yp, hog_slow, \
                                                                                          hog_no_player, \
                                                                                          hog_sec)
########    SERVING POSITION
    
    morpho, fgbg, counter, neg_count, position, positionR, start_flag = firstf.find(args.outdoor, frame, fgbg,\
                                                                      min_area, \
                                                                      max_area, \
                                                                      min_sn_area,\
                                                                      max_sn_area, \
                                                                      mask, mask1, \
                                                                      yc, xc, \
                                                                      kernel, \
                                                                      counter, \
                                                                      neg_count, \
                                                                      counter_thd, \
                                                                      position, \
                                                                      positionR)
########    RECEIVING POSITION
    
    serv_flag, serv_position, serv_positionR, serv_counter, serv_neg_count = f_s_p(morpho, \
                                                                                   xc, yc, \
                                                                                   serv_position, \
                                                                                   serv_positionR, \
                                                                                   serv_counter, \
                                                                                   serv_neg_count)
########    BALL SEARCH
    
    end_flag, end_counter, init_frame = endf(end_counter, init_frame, frame, fps, border,\
                                             min_ball_area, max_ball_area, min_dif_border, \
                                             max_dif_border, greenLower, greenUpper, \
                                             canny_thr)  # find an end frame

########    FLAG IDENTIFICATION
    if curr == 0:
        if start_flag > 0:
            D.append(1)
            curr = 1
        elif hog_flag > 0.5:
            D.append(5)
            curr = 5
        elif serv_flag > 0:
            D.append(4)
            curr = 4
        else:
            D.append(0)
            curr = 0
    else:
        if curr == 1 or curr == 5:
           D.append(2)
           curr = 2
           end_counter = 0
        elif curr == 4:
           D.append(2)
           curr = 2
           end_counter = -5*fps
        else:
            if end_flag > 0:
                D.append(3)
                curr = 0
            elif hog_flag < 0.5:
                D.append(6)
                curr = 0
            else:
                D.append(2)
                curr = 2

        
####### Reinit the in video in order to play it again and edit flags
                
cap.release()
cap = cv2.VideoCapture(video)
for t in range(args.start*fps):
    ret, frame = cap.read()
    
####### Edit the flags
    
F = np.copy(D)
for i in range(2*fps, len(D)-2*fps-1): # if it is a start, go back for 2 sec and 10 frames forward
    if D[i] == 1:
        if i < 2*fps:
            for j in range(i+np.uint8(fps/3)):
                F[j] = 1
        else:
            for j in range(i-3*fps,i+np.uint8(fps/3)): # if it is a start 
               if D[j] == 0 or D[j] == 5:
                   F[j] = 1                # and less then a sec left - assign start from 0, not 25 frames back
    elif D[i] == 3: # if it is an end frame - let it stay for a sec
        for j in range(i, i+2*fps):
            if D[j] == 0:
                F[j] = 3
    elif D[i] == 4: 
        for j in range(i-2*fps, i):
            if D[j] == 0:
                F[j] = 4
    elif D[i] == 5: 
        for j in range(i-3*fps, i):
            if D[j] == 0:
                F[j] = 5
    elif D[i] == 6: 
        for j in range(i-8*fps, i-2*fps):
            F[j] = 6

    # Put texts on frames         
    for i in range(len(F)-1):
        ret, frame = cap.read()
        if not(ret):
            break
        if F[i] != 0 and F[i] != 6:
            out.write(frame)
            
    end = timer()
    print("\nProcessing completed successfully!")
    minutes = np.int(np.floor((end - start)/60))
    sec = np.int(np.round((end - start) - np.int(np.floor((end - start)/60)*60)))
    if sec < 10:
        sec = "0"+str(sec)
    if minutes < 10:
        print("Total time: ", "0"+str(minutes)+":"+str(sec))
    else:
        print("Total time: ", str(minutes)+":"+str(sec))
    out.release()
    out = None
    cap.release()
    cap = None
