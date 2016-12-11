import cv2
import numpy as np
from argparse import ArgumentParser
from sys import argv
from timeit import default_timer as timer
##from matplotlib import pyplot as plt
import find_start_frame_v2 as firstf
from ball_tracking_v2 import fnd_endframe as endf
from court_det import find_serv_point as f_sp
from service_find import find_service_pos as f_s_p

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
parser.add_argument('-c', type=bool, dest='cut',
                  action='store', default=False,
                  help="Cut the video?")
args = parser.parse_args()

start = timer()
D = []

# Init video in and out video
##video = "D:\Waterloo tennis Rodrigo 15_06_2016.mp4"
##fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
video = args.fname #"D:\Waterloo tennis Rodrigo 15_06_2016.mp4"
cap = cv2.VideoCapture(video)
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS ))
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter("processed.avi", fourcc, np.double(fps), (Width,Height), True)
print(out.isOpened()) # Check if file is sucessfully created
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
    5: 'SERVICE POSITION'
} # Dictionary for flags

## Start frame flag search parameters-------------------------------
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
max_sec = 3.0
greenLower = (15, 100, 50)
greenUpper = (52, 150, 255)
canny_thr = 100
border = max_sec*fps
end_counter = 0
found_max_sec = 2
thresh_ball_fond = found_max_sec*fps
ball_cont_found = 0
##------------------------------------------------------------------

## Do the first iteration in order to obtain init_frame for end_frame search
ret, frame = cap.read()
loc_frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
(xc,yc) = f_sp(loc_frame)
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
# Process the video
while(1):
    ret, frame = cap.read()
    if not(ret): # stop if it the end of the video
        break
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
                                                                      positionR) # find a service frame
    
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
    if end_counter == 0:
        f +=1
        if f > thresh_ball_fond:
            ball_cont_found = 1
        else:
            ball_cont_found = 0
    else:
        f = 0
##    print(counter)
    if start_flag > 0:
        if curr == 2:
            for e in range((len(D)-fps),(len(D)-1)):
                if D[e] != 2:
                    break
                else:
                    D[e] = 0
        D.append(1)
        curr = 1
    else:
        if curr == 1:
           D.append(2)
           curr = 2
           end_counter = 0
        else:
            if curr == 2 and end_flag > 0:
                D.append(3)
                curr = 3
            elif curr == 2 and ball_cont_found > 0:
                D.append(4)
                curr = 4
            elif curr == 2:
                D.append(2)
                curr = 2
            elif curr == 0 and serv_flag > 0:
                D.append(5)
                curr = 1
            else:
                D.append(0)
                curr = 0
        
# Reinit the in video in order to play it again and edit flags
cap.release()
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
            for j in range(i-2*fps,i+np.uint8(fps/3)): # if it is a start 
               if D[j] == 0:
                   F[j] = 1                # and less then a sec left - assign start from 0, not 25 frames back
    elif D[i] == 3: # if it is an end frame - let it stay for a sec
        for j in range(i, i+2*fps):
            if D[j] == 0:
                F[j] = 3
    elif D[i] == 4: # if it is an end frame - let it stay for a sec
        for j in range(i, i+2*fps):
            if D[j] == 0:
                F[j] = 4
    elif D[i] == 5: # if it is an end frame - let it stay for a sec
        for j in range(i-2*fps, i):
            if D[j] == 0:
                F[j] = 5

# Put texts on frames         
for i in range(len(F)-1):
    ret, frame = cap.read()
    if not(ret):
        break
    if args.cut:
        if F[i] > 0:
            text = text_to_put[F[i]]
            if F[i] != 2:     
                cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
                cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 4)
            out.write(frame)
    else:
        text = text_to_put[F[i]]   
        cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 4)
        out.write(frame)
        

end = timer()
print("Total time: ", end - start) 
out.release()
out = None
cap.release()
cap = None
