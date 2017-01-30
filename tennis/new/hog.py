from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2
from timeit import default_timer as timer
start = timer()

name = 'tocut_video.avi'
cap = cv2.VideoCapture('/media/pc/ntfs/downloads/'+name)
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter("hog-"+name+".avi", fourcc, np.double(30), (Width,Height), True)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
i = 0
while(1):
    ret, frame = cap.read()
    if not(ret):
        break
    i+=1
    print(i)
    image = cv2.resize(frame,None,fx=0.5, fy=0.5,
                           interpolation = cv2.INTER_CUBIC)
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
    for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    out.write(image)

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
