import cv2
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import find_start_frame as firstf
from ball_tracking import fnd_endframe as endf

start = timer()
# Init video in and out video
fps = 30
video = "D:\Waterloo tennis Rodrigo 15_06_2016.mp4"
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter("new.avi", fourcc, np.double(fps), (1280,720), True)
print(out.isOpened()) # Check if file is sucessfully created
cap = cv2.VideoCapture(video)
for t in range(17250):
    ret, frame = cap.read()
##rr,fr = cap.read()
##fr = cv2.resize(fr,None,fx=0.5, fy=0.5,
##                           interpolation = cv2.INTER_CUBIC)
##cv2.imwrite("olll.jpg", fr)
##ff
# Create foreground extracter
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
D = [] # Array of flags
text_to_put = {
    0: 'IDL',
    1: 'SERVICE',
    2: 'GAME',
    3: 'END',
} # Dictionary for flags


# Process the video
while(1):
    ret, D = firstf.find(cap,fgbg,D) # find a service frame
    if not(ret): # stop if it the end of the video
        break
    print('Move')
    ret, D = endf(cap, fps, D)  # find an end frame
    if not(ret): # stop if it the end of the video
        break

# Reinit the in video in order to play it again and edit flags
cap.release()
cap = cv2.VideoCapture(video)
for t in range(17250):
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
        for j in range(i, i+fps):
            if D[j] == 0:
                F[j] = 3

# Put texts on frames         
for i in range(len(F)-1):
    ret, frame = cap.read()
    text = text_to_put[F[i]]
    cv2.rectangle(frame, (0,0), (180, 60), (0,0,0), -1)
    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 4)
    out.write(frame)

end = timer()
print(end - start) 
out.release()
cap.release()
