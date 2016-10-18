import cv2
import numpy as np
import imutils

video = cv2.VideoCapture('Guillaume et Thomas test deux cameras en bas 1.mp4')
min_area = 100
firstFrame = None

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_,cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    firstFrame = gray
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(10) & 0xFF
        if key  == ord('q'):
            break
    
 
video.release()
cv2.destroyAllWindows()
