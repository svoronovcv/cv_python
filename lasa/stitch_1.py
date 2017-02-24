import cv2
import numpy as np
import segm_1 as segm
from timeit import default_timer as timer

def unsharp(image):
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    return image

def ROIfind(X,Y,side):
    xm = X[i]-side
    xp = X[i]+side
    ym = Y[i]-side
    yp = Y[i]+side
    if X[i] < side:
        xm = 0
    if X[i]+side > 384:
        xp = 384
    if Y[i] < side:
        ym = 0
    if Y[i]+side > 470:
        yp = 470
    return xm,xp,ym,yp

def equal(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
##    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
##    img_yuv = img
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def imgform(img):
    img1 = img[:286,:,:]
    img2 = img[392:,:,:]
    tow = np.vstack((img1,img2))
    img_output = equal(tow)
    tow = cv2.cvtColor(img_output, cv2.COLOR_BGR2LAB)
    return tow
    
def rfile(file):
    A = []
    B = []
    for line in file:
        lst = [eval(i) for i in line.split()]
        A.append(int(lst[1]))
        B.append(int(lst[2]))
    return A, B

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter('ff_7.avi', fourcc, 15.0, (384,470), True)
cap = cv2.VideoCapture('full.avi')
##cap2 = cv2.VideoCapture('fulll.avi.output.avi')
f = open("fulll.avi.outtext.txt", 'r')
side = 50
X,Y = rfile(f)

i=0

while(i < 2):
    ret, frame = cap.read()
    if not(ret):
        break
    i+=1

start = timer()
N = len(X)-3

##while(i < 17040):
##    ret, frame = cap.read()
##    i+=1

while(i < N+2): #len(X)):    
    ret, img = cap.read()
    if not(ret):
        break
    tow = imgform(img)
##    tow = unsharp(tow)
    xm,xp,ym,yp = ROIfind(X,Y,side)
    segm.segment(tow[ym:yp, xm:xp, :], side) #(tow[287:,:,:]) # 
    cv2.circle(tow, (X[i], Y[i]),5, (255,0,0),thickness=-1)
    i+=1
    print(i)
    tow = cv2.cvtColor(tow, cv2.COLOR_LAB2BGR)
    out.write(tow)
##    cv2.imshow("images",tow)
##    k = cv2.waitKey() & 0xff
##    if k == 27:
##        break
##    k = cv2.waitKey(100) & 0xff
##    if k == 27:
##        break
##cv2.imwrite("st.jpg", tow)
    
end = timer()
print(end - start)
print((end - start) / N)
cv2.destroyAllWindows()
cap.release()
##cap2.release()
out.release()
out = None
cap = None
##cap2 = None

