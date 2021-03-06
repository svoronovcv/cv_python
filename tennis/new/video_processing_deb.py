import cv2
import numpy as np
import os
import sys
from argparse import ArgumentParser
from sys import argv
from timeit import default_timer as timer
##from matplotlib import pyplot as plt
import find_start_frame_v2_debug as firstf
from ball_tracking_v2 import fnd_endframe as endf
from court_det import find_serv_point as f_sp
from service_find_debug import find_service_pos as f_s_p
from player_speed_debug import speed

def arg2bool(arg):
    if arg == "True" or arg == True:
        return True
    else:
        return False

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
parser.add_argument('-d', type=str, dest='debug',
                  action='store', default=False,
                  help="Debug the video?")
args = parser.parse_args()
start = timer()
D = []

# Init video in and out video
##video = "D:\Waterloo tennis Rodrigo 15_06_2016.mp4"
##fourcc = cv2.VideoWriter_fourcc('F','M','P','4')

args.cut = arg2bool(args.cut)
args.debug = arg2bool(args.debug)
video = args.fname #"D:\Waterloo tennis Rodrigo 15_06_2016.mp4"
cap = cv2.VideoCapture(video)
frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS ))
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
if not(args.debug):
    out = cv2.VideoWriter("processed-"+os.path.basename(video)+".avi", fourcc, np.double(fps), (Width,Height), True)
else:
    out = cv2.VideoWriter("debug-"+os.path.basename(video)+".avi", fourcc, np.double(fps), (Width,int(Height/2)), True)
    outmor = cv2.VideoWriter("mor-"+os.path.basename(video)+".avi", fourcc, np.double(fps), (int(Width/2),int(Height/2)), True)
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
    4: 'CONTINUOUS BALL',
    5: 'SERVICE POSITION',
    6: 'HIGH SPEED',
    7: 'SLOW PLAYER'
} # Dictionary for flags

## Start frame flag search parameters-------------------------------
sp_ar = []
ball_arr = []
mor_arr = []
pos_arr = []
plrec_arr = []
min_area = 80
max_area = 4000
min_sn_area = 10
max_sn_area = 200
mask = np.zeros((int(Height/2), int(Width/2),3), dtype=np.uint16)
mask[100:, :,:] = 1
mask1 = np.zeros((int(Height/2), int(Width/2),3), dtype=np.uint16)+255 #cv2.imread('ff.jpg')
##(yc, xc) = (Height/4 + 100 -40*args.height, Width/4)
##(yc, xc) = (215, 307)
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

## Do the first iteration in order to obtain init_frame for end_frame search
ret, frame = cap.read()
loc_frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
(xc,yc) = (318, 206) #f_sp(loc_frame)
cv2.imwrite('n2.jpg',loc_frame)
print(xc,yc)

##cv2.imwrite('dfg.jpg', loc_frame)
morpho, fgbg, counter, neg_count, position, positionR, start_flag, morka = firstf.find(args.outdoor, frame, fgbg, \
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
M = np.uint8(np.zeros(frame.shape))
M = cv2.resize(M,None,fx=0.5, fy=0.5,
                   interpolation = cv2.INTER_CUBIC)
M[:,:,0] = np.uint8(morka)
M[:,:,1] = np.uint8(morka)
M[:,:,2] = np.uint8(morka)
outmor.write(M)
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
print("The video file is being processed:")
while(1):
    ret, frame = cap.read()
##    cv2.imshow('ola', frame)
##    cv2.waitKey(10)
    it += 1
##    print(it)
    if not(ret): # stop if it the end of the video
        break
    procent = np.round(1000*it/frames_num)/10
    if procent > 100:
        procent = 100
    sys.stdout.write('\r%s %%' % procent)
    sys.stdout.flush()
    morpho, fgbg, counter, neg_count, position, positionR, start_flag, morka = firstf.find(args.outdoor, frame, fgbg,\
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
    M = np.uint8(np.zeros(frame.shape))
    M = cv2.resize(M,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
    M[:,:,0] = np.uint8(morka)
    M[:,:,1] = np.uint8(morka)
    M[:,:,2] = np.uint8(morka)
    outmor.write(M)
    
    serv_flag, serv_position, serv_positionR, serv_counter, serv_neg_count = f_s_p(morpho, \
                                                                                   xc, yc, \
                                                                                   serv_position, \
                                                                                   serv_positionR, \
                                                                                   serv_counter, \
                                                                                   serv_neg_count)
    end_flag, end_counter, init_frame = endf(end_counter, init_frame, frame, fps, border,\
                                             min_ball_area, max_ball_area, min_dif_border, \
                                             max_dif_border, greenLower, greenUpper, \
                                             canny_thr)  # find an end frame
    if it > 150:
        speed_start, slow, slow_count, fast_count, speed_position, no_player, very_slow, speedushka, rectushka = speed(speed_start, \
                                                                                     morpho, slow, slow_count, \
                                                                                     fast_count, speed_position, \
                                                                                     no_player, very_slow)
        sp_ar.append(speedushka)
        pos_arr.append(speed_position)
        plrec_arr.append(rectushka)
    else:
        pos_arr.append((0,0))
        sp_ar.append(0)
        plrec_arr.append((0,0,0,0))
    
        if speed_start < 1:
            high_speed +=1
        else:
            if high_speed > fps:
                speed_flag = 1
            else:
                speed_flag = 0
            high_speed = 0
    if end_counter == 0:
        ball_arr.append('BALL')
        f +=1
        if f > thresh_ball_fond:
            ball_cont_found = 1
        else:
            ball_cont_found = 0
    else:
        nf_counter += 1
        ball_arr.append('NO BALL')
    if nf_counter > thresh_ball_nf:
        f = 0
##    print(counter)
    if very_slow > 4*fps:
        slow_flag = 1
        very_slow = 0
    else:
        slow_flag = 0
    if start_flag > 0:
        very_slow = -5*fps
        if curr == 2:
            for e in range((len(D)-fps),(len(D)-1)):
                if D[e] != 2:
                    break
                else:
                    D[e] = 0
        D.append(1)
        curr = 1
    else:
        if speed_flag > 0 and curr == 0:
            very_slow = -5*fps
            D.append(6)
            curr = 5
        if curr == 1:
           D.append(2)
           curr = 2
           end_counter = 0
        elif curr == 5:
           D.append(2)
           curr = 2
           end_counter = -5*fps
        else:
            if curr == 2 and end_flag > 0:
                D.append(3)
                curr = 3
            elif curr == 2 and ball_cont_found > 0:
                D.append(4)
                curr = 4
            elif curr == 2 and slow_flag > 0:
                D.append(7)
                curr = 3
            elif curr == 2:
                D.append(2)
                curr = 2
            elif curr == 0 and serv_flag > 0:
                D.append(5)
                very_slow = -5*fps
                curr = 5
            else:
                D.append(0)
                curr = 0
        
# Reinit the in video in order to play it again and edit flags
cap.release()
capmor = cv2.VideoCapture("mor-"+os.path.basename(video)+".avi")
cap = cv2.VideoCapture(video)
for t in range(args.start*fps):
    ret, frame = cap.read()
# Edit the flags
F = np.copy(D)
for i in range(2*fps, len(D)-fps-1): # if it is a start, go back for 2 sec and 10 frames forward
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
    elif D[i] == 4: # if it is an end frame - let it stay for a sec
        for j in range(i, i+2*fps):
            if D[j] == 0:
                F[j] = 4
    elif D[i] == 5: 
        for j in range(i-2*fps, i):
            if D[j] == 0:
                F[j] = 5
    elif D[i] == 6: 
        for j in range(i-4*fps, i):
            if D[j] == 0:
                F[j] = 6
    elif D[i] == 7: 
        for j in range(i-fps, i):
            if D[j] == 2:
                F[j] = 7
            else:
                break

# Put texts on frames         
for i in range(len(F)-1):
    ret, frame = cap.read()
    retmor, mor = capmor.read()
    if not(ret) or not(retmor):
        break
    if not(args.debug):
        if args.cut:
            if F[i] > 0:
    ##            text = text_to_put[F[i]]
    ##            if F[i] != 2:     
    ##                cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
    ##                cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
    ##                            1.0, (255, 255, 255), 4)
                out.write(frame)
        else:
            text = text_to_put[F[i]]   
            cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 4)
            out.write(frame)
    else:
        frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
        text = text_to_put[F[i]]   
        cv2.rectangle(frame, (0,0), (200, 90), (0,0,0), -1)
        cv2.putText(frame, text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 3)
        if sp_ar[i] > 6:
            col = (0,0,255)
        elif sp_ar[i] < 1:
            col = (255,0,0)
        sp_text = str(int(sp_ar[i]))   
        cv2.putText(frame, sp_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, col, 3)
        ball_text = str(ball_arr[i])
        cv2.putText(frame, ball_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 3)
        M = mor
##        M = np.uint8(np.zeros(frame.shape))
##        M[:,:,0] = np.uint8(mor_arr[i])
##        M[:,:,1] = np.uint8(mor_arr[i])
##        M[:,:,2] = np.uint8(mor_arr[i])
        if i >151:
            cv2.circle(frame, tuple(pos_arr[i]), 3, (255,0,0), -1)
            cv2.circle(M, tuple(pos_arr[i]), 3, (255,0,0), -1)
            cv2.circle(frame, tuple(pos_arr[i-1]), 3, (0,255,0), -1)
            cv2.circle(M, tuple(pos_arr[i-1]), 3, (0,255,0), -1)
            cv2.line(M, tuple(pos_arr[i-1]), tuple(pos_arr[i]), (0,0,255), 3)
            cv2.line(frame, tuple(pos_arr[i-1]), tuple(pos_arr[i]), (0,0,255), 3)
            cv2.rectangle(frame, (plrec_arr[i][0], plrec_arr[i][1]),(plrec_arr[i][0]+plrec_arr[i][2],plrec_arr[i][1]+plrec_arr[i][3]), (255, 0, 0), 2)
            cv2.rectangle(M, (plrec_arr[i][0], plrec_arr[i][1]),(plrec_arr[i][0]+plrec_arr[i][2],plrec_arr[i][1]+plrec_arr[i][3]), (255, 0, 0), 2)
        frame = np.hstack((frame,M))
##        cv2.imshow('ola', frame)
##        cv2.waitKey()
        cv2.imwrite('n2.jpg',frame)
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
outmor.release()
outmor = None
cap.release()
cap = None
cv2.destroyAllWindows()
