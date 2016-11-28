import cv2
import numpy as np

def find_serv_point(img):
    stepy = 16
    stepx = 6
    mask = np.zeros((stepy,stepx), dtype=np.uint8)
    mask[7:9:,] = 1
    mask[4:7,2:4] = 1
    mask_2 = np.zeros(np.shape(img[:,:,0]), dtype=np.uint8)
    stx = np.int16(3*np.shape(img)[1]/8)
    enx = np.int16(5*np.shape(img)[1]/8)
    sty = np.int16(np.shape(img)[0]/2)
    eny = np.int16(np.shape(img)[0] - stepy-1)
    mask_2[sty:eny, stx:enx] = 1
    Ipart = (img[:,:,2]>150)*(img[:,:,1]>150)*(img[:,:,0]>100)*mask_2
    maxs=0
    mi = 0
    mj = 0
    stepicx = np.uint8(stepx/2)
    stepicy = np.uint8(stepy/2)
    for x in range(stx,enx):
        for y in range(sty,eny):
            Ip = Ipart[y-stepicy:y+stepicy, x-stepicx:x+stepicx]
            s = np.sum(Ip == mask)
            if s > maxs:
                maxs = s
                mi = x
                mj = y

    return mi,mj
