import cv2
import numpy as np

def rfile(file):
    A = []
    for line in file:
        A.append(int(line))
    return A

def fval(num, arr):
    pos = -1
    x = 0
    for i in arr:
        if np.abs(i-num) < 50:
            return x
        x+=1
    return pos

def arrform(arr1, arr2):
    D = []
    for t in arr1:
        D.append(fval(t,arr2))
    return np.array(D)

f1 = open("outdoor.txt", 'r')
f2 = open("out.txt", 'r')

A = rfile(f1)
B = rfile(f2)

FNarr = arrform(A,B)
FParr = arrform(B,A)

print(FNarr, '\n')
print("FN = ", np.sum(FNarr == -1), '\n')
print( FParr, '\n')
print("FP = ", np.sum(FParr == -1), '\n')
print("Precision = ", (FNarr.shape[0] - np.sum(FNarr == -1)) / ((FNarr.shape[0] - np.sum(FNarr == -1)) + np.sum(FParr == -1)), '\n')
print("Recall = ", (FNarr.shape[0] - np.sum(FNarr == -1)) / ((FNarr.shape[0] - np.sum(FNarr == -1)) + np.sum(FNarr == -1)), '\n')


f1.close()
f2.close()



