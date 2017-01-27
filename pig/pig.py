import cv2
import numpy as np
import time
import pymeanshift as pms


def segment(img):
    shifted = cv2.pyrMeanShiftFiltering(img, 100, 100)
    ##shifted = cv2.pyrMeanShiftFiltering(ROI, sp=7, sr=25, \
    ##                                    maxLevel=1, \
    ##                                    termcrit=(
    ##                                        cv2.TERM_CRITERIA_EPS \
    ##                                        + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
##    cv2.imshow('olaa', shifted)
##    cv2.waitKey()
    (seg_image, lab_image, num_regions) = pms.segment(shifted, \
                                                      spatial_radius=5,\
                                                      range_radius=5, \
                                                      min_density=50)
    for i in range(num_regions):
        A = np.uint8(lab_image == i)*200
##        cv2.imshow('seg', A)
##        cv2.waitKey()
        (_,cnts,_) = cv2.findContours(A, cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if (cv2.contourArea(c) < 1000) or (cv2.contourArea(c) > 100000):
                continue
            epsilon = 0.01*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(img,[approx],0,(255,0,0),2)
    return num_regions

cap = cv2.VideoCapture('Pig 1.MOV')
frame = None
for i  in range(90):
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_CUBIC)
Width = frame.shape[1]
Height = frame.shape[0]
fps = int(cap.get(cv2.CAP_PROP_FPS ))
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter("processed-pig.avi", fourcc, np.double(25), (Width,Height), True)
i = 0
while(i < 25):
    ret, frame = cap.read()
    i+=1
    print(i)
    if not(ret):
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,interpolation = cv2.INTER_CUBIC)
    A = segment(frame)
    out.write(frame)

out.release()
out = None
##rows,cols = frame.shape[0], frame.shape[1]
##M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
##dst = cv2.warpAffine(frame,M,(cols,rows))

##            (x, y, w, h) = cv2.boundingRect(c)
##            B.append((x, y, w, h))
    
##    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in B])
##    pick = non_max_suppression(rects, overlapThresh=0.40)
##    for (xA, yA, xB, yB) in rects:  
##            cv2.rectangle(ROI, (xA, yA), (xB, yB), (0, 255, 0), 2)
##    cv2.imshow("Thresh", ROI)
##    cv2.imshow("Segmented", seg_image)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

##frame = cv2.imread("st.jpg")
##ROI = frame[310:465, 45:355, :]
##gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
##thresh = 255-cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
##   cv2.THRESH_BINARY,11,3)#cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
##
##kernel = np.ones((3,3),np.uint8)
##opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
##
##sure_bg = cv2.dilate(thresh,kernel,iterations=3)
##
##dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
##ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
##
##sure_fg = np.uint8(sure_fg)
##unknown = cv2.subtract(sure_bg,sure_fg)
##
##ret, markers = cv2.connectedComponents(sure_fg)
##
##markers = markers+1
##
##markers[unknown==255] = 0
##startT = time.time()
##shifted = cv2.pyrMeanShiftFiltering(ROI, 11, 41)
####shifted = cv2.pyrMeanShiftFiltering(ROI, sp=11, sr=41, \
####                                    maxLevel=1, \
####                                    termcrit=(
####                                        cv2.TERM_CRITERIA_EPS \
####                                        + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
##(seg_image, lab_image, num_regions) = pms.segment(shifted, \
##                                                  spatial_radius=2,\
##                                                  range_radius=2, \
##                                                  min_density=50)
####markers = cv2.watershed(ROI,markers)
####ROI[markers == -1] = [255,0,0]
##endt = time.time()
##print(num_regions)
##B = []
##for i in range(num_regions):
##    A = np.uint8(lab_image == i)*200
##    (_,cnts,_) = cv2.findContours(A, cv2.RETR_CCOMP,
##		cv2.CHAIN_APPROX_SIMPLE)
##    for c in cnts:
##        if (cv2.contourArea(c) < 50) or (cv2.contourArea(c) > 5000):
##            continue
##        (x, y, w, h) = cv2.boundingRect(c)
##        B.append((x, y, w, h))
##print(len(B))
##rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in B])
##pick = non_max_suppression(rects, overlapThresh=0.40)
##print(len(pick))
##for (xA, yA, xB, yB) in pick:
##	cv2.rectangle(ROI, (xA, yA), (xB, yB), (0, 255, 0), 2)
##print(endt-startT)
##cv2.imshow("Thresh", ROI)
##cv2.imshow("Segmented", seg_image)
####cv2.imshow("binary", thresh)
####cv2.imshow("opening", opening)
####cv2.imshow("sure bg", sure_bg)
####cv2.imshow("sure fg", sure_fg)
####cv2.imshow("markers", markers)
####cv2.imshow("image", shifted)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
