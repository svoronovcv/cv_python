from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2
from timeit import default_timer as timer
start = timer()

name = 'Guillaume et Thomas test deux cameras en bas 1.mp4'
cap = cv2.VideoCapture(name)
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-200
fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
out = cv2.VideoWriter("debhog-"+name+".avi", fourcc, np.double(3), (Width,Height), True)
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
i = 0
zer = 0
mnogo = 0
to = 0
proplay = (0, 0, 0, 0)
speed = 0
xp , yp = 0, 0
xpp , ypp = 0, 0
no_player = 1
slow=0
sec = 10
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
    print(i)
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
    pick = rects #non_max_suppression(rects, probs=None, overlapThresh=0.65)
##    if len(pick) < 1:
##        no_player += 1
##        speed = 0
##    else:
##        no_player = 1
    # draw the final bounding boxes
    j = 0
    mind = 120000000
    xc = image.shape[1]/2
    yc = image.shape[0]
    cv2.circle(image, (int(xpp),int(ypp)), 5, (255,0,0), -1)
    for (xA, yA, xB, yB) in pick:
        if (xB - xA) < 50 or (yB - yA) < 75 or ((xB - xA) < 150 and yB > 3*image.shape[0]/4) or yB < image.shape[0]/6:
            continue
        j+=1
        dst = min((xA-xc)**2 + (yB-yc)**2,(xB-xc)**2 + (yB-yc)**2)
        if dst < mind:
            if dst + 150 < mind:
                proplay = (xA, yA, xB, yB)
                xp = (xA+xB)/2
                yp = (yA+yB)/2
                mind = dst
            else:
                if yB > yp:
                    proplay = (xA, yA, xB, yB)
                    xp = (xA+xB)/2
                    yp = (yA+yB)/2
                    mind = dst
    speed = np.sqrt((xp-xpp)**2 + (yp-ypp)**2) / no_player
    if j < 1:
        no_player += 1
        speed = 0
    else:
        no_player = 1
    xpp = xp
    ypp = yp
    if j < 1 or speed > 200:
        speed = 0
    if speed < 30:
        slow+=1
    else:
        slow = 0
    if no_player > 3*sec or slow > 3*sec:
        text = "slow"
    else:
        text = "G"
##    text = str(speed)
    cv2.circle(image, (int(xpp),int(ypp)), 5, (0,0,255), -1)
    cv2.rectangle(image, (0,0), (400, 60), (0,0,0), -1)
    cv2.putText(image, text, (30, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.putText(image, str(speed), (00, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.putText(image, str(no_player), (00, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.rectangle(image, (proplay[0], proplay[1]), (proplay[2], proplay[3]), (0, 255, 0), 2)
    out.write(image)
##    if speed > 100:
##        cv2.imshow('image', image)
##        cv2.waitKey()
##    if j !=1:
##    cv2.imshow('image', image)
##    cv2.waitKey()
##    k = cv2.waitKey(1) & 0xff
##    if k == ord('q'):
##        break
##        out.write(image)
##        if j < 1:
##            zer+=1
##        else:
##            mnogo+=1
##            text = text_to_put[F[i]]
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
