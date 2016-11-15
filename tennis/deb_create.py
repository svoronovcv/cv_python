import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('/media/pc/ntfs/downloads/Guillaume vs Thomas test deux cameras 2 en haut.mp4')
f = open('Guillaume vs Thomas test deux cameras 2 en haut.txt', 'w')

fr_number = 0
##for i in range(5700):
##    ret, frame = cap.read()
##    fr_number += 1
##    
##frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
##                       interpolation = cv2.INTER_CUBIC)
while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    fr_number += 1
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(50) & 0xff
    if k == ord('q'):
        plt.imshow(frame)
        plt.show()
        f.write(str(fr_number) + '\n')
##    k = cv2.waitKey(50) & 0xff
##    if k == 27:
##        break
    
cap.release()
f.close()
cv2.destroyAllWindows()
