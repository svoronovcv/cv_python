import cv2
import numpy as np
import segm

cap = cv2.VideoCapture('full.avi') # video names
cap2 = cv2.VideoCapture('fulll.avi.output.avi')
i=0
while(i < 2):
    ret, frame = cap.read()
    if not(ret):
        break
    i+=1

##while(i < 30000):
##   ret, frame = cap.read()
##   ret, frame = cap.read()
##    i+=1
while(1):    
    ret, img = cap.read()
    img1 = img[:286,:,:] #divide upper and lower images 
    img2 = img[392:,:,:]
    tow = np.vstack((img1,img2)) # stack images together
    segm.segment(tow[:, :, :]) # perform segmentation
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
##cv2.imwrite("st.jpg", tow)
cv2.destroyAllWindows()
cap.release()
cap2.release()
cap = None
cap2 = None
