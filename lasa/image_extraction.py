import cv2
import numpy as np
import os
from glob import glob

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

# get video names
video_names = glob('Exp4/*.avi')
# image size: width, height
w = 384
h = 144
# camera calibration parameters
camera_matrix= np.array([[263.3884, -0.7857, 175.7669],[0,247.6750, 120.0455],[0,0,1]])
dst_coeffs= np.array([[-0.4026,0.1329,0.0027,0.00036523,0]])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dst_coeffs, (w, h), 1, (w, h))

for vn in video_names:
    path, name, ext = splitfn(vn)
    fold = path+'/'+name
    if not os.path.isdir(fold):
        os.mkdir(fold)
    cap = cv2.VideoCapture(vn)
    i = 0
    while(1):
        ret, frame = cap.read() # read a frame
        if not(ret):
            break
        i+=1
        img = frame[:,:,:]
		# undistort the image
        dst = cv2.undistort(img, camera_matrix, dst_coeffs, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
		# save the image
        cv2.imwrite(fold+'/'+str(i)+'-undist'+'.jpg', dst)
		# save the initial image
        cv2.imwrite(fold+'/'+str(i)+'.jpg', img)
