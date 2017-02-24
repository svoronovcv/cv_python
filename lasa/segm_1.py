import cv2
import numpy as np
import time
import pymeanshift as pms


def find_if_close(cnt1,cnt2, max_dist):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < max_dist :
                return True
            elif i==row1-1 and j==row2-1:
                return False
        
def segment(ROI,side):
    shifted = cv2.pyrMeanShiftFiltering(ROI, 7, 31)
    ##shifted = cv2.pyrMeanShiftFiltering(ROI, sp=7, sr=25, \
    ##                                    maxLevel=1, \
    ##                                    termcrit=(
    ##                                        cv2.TERM_CRITERIA_EPS \
    ##                                        + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
    (seg_image, lab_image, num_regions) = pms.segment(shifted, \
                                                      spatial_radius=3,\
                                                      range_radius=3, \
                                                      min_density=0)
    B = []
    min_dist = 10000
    gaze_c = []
    for i in range(num_regions):
        A = np.uint8(lab_image == i)*200
        (_,cnts,_) = cv2.findContours(A, cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if (cv2.contourArea(c) < 50) or (cv2.contourArea(c) > 3000):
                continue
            d = cv2.pointPolygonTest(c,(side,side),True)*(-1)
            if d < min_dist:
                min_dist = d
                gaze_c = c
##            epsilon = 0.01*cv2.arcLength(c,True)
##            approx = cv2.approxPolyDP(c,epsilon,True)
##            cv2.drawContours(ROI,[approx],0,(100,100,100),2)
    if gaze_c != []:
            epsilon_g = 0.01*cv2.arcLength(gaze_c,True)
            approx_g = cv2.approxPolyDP(gaze_c,epsilon_g,True)
            cv2.drawContours(ROI,[approx_g],0,(150,150,150),2)


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
