import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('/media/pc/ntfs/downloads/Guillaume et Thomas test deux cameras en bas 1.mp4')
mask = np.zeros((360,640,3), dtype=np.uint8)
mask[:, :,:] = 1

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
##    frame = frame*mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    abs_sobel = np.absolute((sobelx)**2 + (sobely)**2)
    sobel = np.uint8(np.sqrt(abs_sobel))
    mask1 = np.ones(gray.shape, dtype = np.uint8)*(abs_sobel > 1000000)
    mask = np.uint8(mask1)
    (_,cnts,_) = cv2.findContours(mask1, cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if (cv2.contourArea(c) < 100) or (cv2.contourArea(c) > 5000):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Sobel", mask)
    cv2.imshow("Sobel1", mask1)
##    cv2.imshow("Sobel", thresh1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()


