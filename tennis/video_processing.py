import cv2
import numpy as np
from matplotlib import pyplot as plt
import find_start_frame as firstf
import find_end_frame as endf

video = 'Guillaume et Thomas test deux cameras en haut 1.mp4'
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('Guillaume et Thomas test deux cameras en haut 1.avi', fourcc, 30.0, (1280,720), True)
cap = cv2.VideoCapture(video)
fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=200)
D = []
text_to_put = {
    0: 'NOTHING',
    1: 'SERVICE',
    2: 'GAME",
    3: 'END',
}
k = 0
while(1):
    ret, frame, D, k = firstf.find(cap,fgbg,D, k)
    ret, D, k = endf.find(frame,cap,fgbg,D, k)
    if ret:
        break
    
cap.release()
cap = cv2.VideoCapture(video)
for i in range(25, len(D)-26):
    if D[i] = 1:
        if i < 25:
            for j in range(i+10):
                D[j] = 1
        else:
            for j in range(i-25, i+10):
            D[j] = 1
    elif D[i] = 3:
        for j in range(i, i+25):
            D[j] = 3
            
for i in range(len(D)-1):
    ret, frame = cap.read()
    text = text_to_put[D[i]]
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 0, 255), 3)
    out.write(frame)

out.release()
cap.release()


    
    
    
