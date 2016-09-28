import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1]])
img = cv2.imread('1.jpg', 0)
k3 = ndimage.convolve(img, kernel_3x3)
blurred = cv2.GaussianBlur(img, (11,11), 0)
g_hpf = img - blurred
cv2.imshow('3x3', k3)
cv2.imshow('g_hpf', g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()