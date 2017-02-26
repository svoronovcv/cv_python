import cv2
import numpy as np
import os
import sys
from argparse import ArgumentParser
from sys import argv
from timeit import default_timer as timer
##from matplotlib import pyplot as plt
import find_start_frame_v2 as firstf
from ball_tracking_v2 import fnd_endframe as endf
from court_det import find_serv_point as f_sp
from service_find import find_service_pos as f_s_p
from player_speed import speed
from hog_proc import hog_proc

cv2.ocl.setUseOpenCL(False)

parser = ArgumentParser()
parser.add_argument('-n', required = True, dest='fname',
                  action='store', help="Videofile name")
##parser.add_argument('-d', type=int, dest='height',
##                  action='store', default=0,
##                  help="Distance to the court: 0-low, 1-middle, 2-high")
parser.add_argument('-i', type=int, dest='outdoor',
                  action='store', default=0,
                  help="1-outdoor videos, 0-indoor")
parser.add_argument('-s', type=int, dest='start',
                  action='store', default=0,
                  help="Set starting time in seconds")
parser.add_argument('-c', type=str, dest='cut',
                  action='store', default=True,
                  help="Cut the video?")
args = parser.parse_args()

start = timer()
D = []

# Init video in and out video
if args.cut == "True" or args.cut == True:
    args.cut = True
else:
    args.cut = False
video = args.fname #"D:\Waterloo tennis Rodrigo 15_06_2016.mp4"
cap = cv2.VideoCapture(video)
frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS ))
fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
out = cv2.VideoWriter("processed-"+os.path.basename(video)+".avi", fourcc, np.double(fps), (Width,Height), True)
if out.isOpened():
    print("Video file has been created!") # Check if file is sucessfully created
for t in range(args.start*fps):
    ret, frame = cap.read()
# Create foreground extracter
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
text_to_put = {
    0: 'IDLE',
    1: 'SERVICE DETECTED',
    2: 'GAME',
    3: 'LOST BALL',
    4: 'SERVICE POSITION',
    5: 'HOG HIGH SPEED',
    6: 'HOG SLOW PLAYER',
    7: 'BGS HIGH SPEED',
    8: 'BGS SLOW PLAYER'
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
hog_fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
hog_to = 0
hog_speed = 0
hog_xp , hog_yp = 0, 0
hog_xpp , hog_ypp = 0, 0
hog_no_player = 1
hog_slow=0
hog_sec = 10
hog_start = 0
hog_stop = 0
##------------------------------------------------------------------

## BGS Speed parameters------------------------------------
high_speed = 0
speed_flag = 0
slow = 0
slow_count = 0
fast_count = 0
speed_position = [0, 0]
no_player = 1
speed_start = 0
very_slow = 0
slow_flag = 0
##------------------------------------------------------------
AAA = []
## Do the first iteration in order to obtain init_frame for end_frame search
ret, frame = cap.read()
##loc_frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
(xc,yc) = (320,200) #f_sp(loc_frame)#
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
    
    hog_fgbg, hog_to, hog_speed, hog_xpp, hog_ypp, hog_xp, hog_yp, hog_slow, hog_no_player, hog_start, hog_stop = hog_proc(hog_fgbg, hog, frame, hog_speed, \
                                                                                          hog_to, hog_xpp, \
                                                                                          hog_ypp, hog_xp, \
                                                                                          hog_yp, hog_slow, \
                                                                                          hog_no_player, \
                                                                                          hog_sec, fps)
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
######## BGS PLAYER SPEED
    if it > 10000000000000000*fps:
        speed_start, slow, slow_count, fast_count, speed_position, no_player, very_slow = speed(speed_start, \
                                                                                     morpho, slow, slow_count, \
                                                                                     fast_count, speed_position, \
                                                                                     no_player, very_slow)
        if speed_start < 1:
            high_speed +=1
        else:
            if high_speed > fps:
                speed_flag = 1
            else:
                speed_flag = 0
            high_speed = 0
    if very_slow > 4*fps:
        slow_flag = 1
        very_slow = 0
    else:
        slow_flag = 0
            
########    FLAG IDENTIFICATION
    if start_flag > 0:
        AAA.append(it)
    if curr == 0:
        if start_flag > 0:
            D.append(1)
            curr = 1
            hog_slow, hog_no_player = 0, 1
            very_slow = -5*fps
        elif hog_start > 0.5:
            D.append(5)
            curr = 5
            very_slow = -5*fps
        elif serv_flag > 0:
            D.append(4)
            curr = 4
            hog_slow, hog_no_player = 0, 1
            very_slow = -5*fps
        elif speed_flag > 0:
            D.append(7)
            curr = 7
            very_slow = -5*fps
        else:
            D.append(0)
            curr = 0
    else:
        if curr == 1 or curr == 5 or curr == 7:
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
            elif hog_stop > 0.5:
                D.append(6)
                curr = 0
            elif slow_flag > 0:
                D.append(8)
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
                F[j] = 1                # and less then a sec left - assign start from 0, not 25 frames back
    elif D[i] == 3: # if it is an end frame - let it stay for a sec
        for j in range(i, i+2*fps):
            if D[j] == 2:
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
        for j in range(i-8*fps, i):
            if D[j] == 2:
                F[j] = 6
    elif D[i] == 7: 
        for j in range(i-4*fps, i):
            if D[j] == 0:
                F[j] = 7
    elif D[i] == 8: 
        for j in range(i-fps, i):
            if D[j] == 2:
                F[j] = 8

####### Put texts on frames
for i in range(len(F)-1):
    ret, frame = cap.read()
    if not(ret):
        break
    if args.cut:
        if F[i] != 0 and F[i] != 6 and F[i] != 3 and F[i] != 8:
            out.write(frame)
    else:
        text = text_to_put[F[i]]   
        cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 4)
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
