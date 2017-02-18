from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2
from timeit import default_timer as timer
start = timer()

name = 'Tennis Franck Palm.mp4'
cap = cv2.VideoCapture('D://Tennis//'+name)
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-200
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter("hog-"+name+".avi", fourcc, np.double(3), (Width,Height), True)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
i = 0
zer = 0
mnogo = 0
to = 0
proplay = (0, 0, 0, 0)
speed = 0
xp , yp = 0, 0
while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    to+=1
    i+=1
    if to < 10:
        continue
    else:
        to = 0
    print(i, to)
    image = frame[200:,:,:] #cv2.resize(frame,None,fx=0.5, fy=0.5,interpolation = cv2.INTER_CUBIC)
##    orig = np.copy(image)
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.1)
    # draw the original bounding boxes
##    for (x, y, w, h) in rects:
##            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    j = 10
    mind = 120000000
    xc = image.shape[1]/2
    yc = image.shape[0]/2
    cv2.circle(image, (int(xp),int(yp)), 5, (255,0,0), -1)
    for (xA, yA, xB, yB) in pick:
        if (xB - xA) < 100 or (yB - yA) < 150:
            continue
        j+=1
        dst = (xA-xc)**2 + (yA-yc)**2
        if dst < mind:
            proplay = (xA, yA, xB, yB)
            speed = np.sqrt(((xA+xB)/2-xp)**2 + ((yA+yB)/2-yp)**2)
            xp = (xA+xB)/2
            yp = (yA+yB)/2
            mind == dst
    text = str(speed)
    cv2.circle(image, (int(xp),int(yp)), 5, (0,0,255), -1)
    cv2.rectangle(image, (0,0), (400, 60), (0,0,0), -1)
    cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 4)
    cv2.rectangle(image, (proplay[0], proplay[1]), (proplay[2], proplay[3]), (0, 255, 0), 2)
    if j !=1:
##        cv2.imshow('image', image)
##        cv2.waitKey()
##        k = cv2.waitKey(1) & 0xff
##        if k == ord('q'):
##            break
        out.write(image)
        if j < 1:
            zer+=1
        else:
            mnogo+=1

cap.release()
out.release()
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
cv2.destroyAllWindows()
print("zero: ", zer, "; mnogo: ", mnogo)
