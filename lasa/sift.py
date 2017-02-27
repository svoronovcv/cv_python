import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    r1 = 3
    thickness = 3
    if color:
        c = color
        i = 0
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
##        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
##dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        end1 = tuple(np.int16(src_pts[i][0]))
        end2 = tuple(np.int16(dst_pts[i][0]) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
        i+=1
##        cv2.circle(new_img, end1, r1, c, -1)
##        cv2.circle(new_img, end2, r1, c, -1)
    
    cv2.imshow('new',new_img)
    cv2.imwrite('/media/pc/ntfs/Untitled Folder/matchm.jpg',new_img)
##    cv2.imwrite('/media/pc/ntfs/Untitled Folder/f1.jpg',new_img[:,:1280,:])
##    cv2.imwrite('/media/pc/ntfs/Untitled Folder/f2.jpg',new_img[:,1280:,:])
    cv2.waitKey()

MIN_MATCH_COUNT = 10

img1c = cv2.imread('Ajung-2-recording 0321.jpg')          # queryImage
img2c = cv2.imread('Ajung-2-recording 0334.jpg')
cv2.imwrite('/media/pc/ntfs/Untitled Folder/raw1.jpg',img1c)
cv2.imwrite('/media/pc/ntfs/Untitled Folder/raw2.jpg',img2c)
img1 = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)
# Define a feature matching algorithm FLANN or bruteforce
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

bf = cv2.BFMatcher() #cv2.NORM_HAMMING, crossCheck=True)
##flann = cv2.FlannBasedMatcher(index_params,search_params)

# ORB Features for image matching
sift = cv2.ORB_create(nfeatures=1000, scaleFactor = 1.2, nlevels=8, WTA_K=4, patchSize=51)
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
##star = cv2.xfeatures2d.StarDetector_create()
##brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
##kp1 = star.detect(img1,None)
##kp1, des1 = brief.compute(img1, kp1)
T = np.eye(3)
M = np.eye(3)

# compute the descriptors with ORB
# draw only keypoints location,not size and orientation
img1d =cv2.drawKeypoints(img1c,kp1,img1,color=(0,255,0), flags=0)
cv2.imshow('dis1', img1d)
cv2.imwrite('/media/pc/ntfs/Untitled Folder/first.jpg',img1d)
cv2.waitKey()
img2d = cv2.drawKeypoints(img2c,kp2,img2,color=(0,255,0), flags=0)
cv2.imshow('dis2', img2d)
cv2.imwrite('/media/pc/ntfs/Untitled Folder/second.jpg',img2d)
cv2.waitKey()

# Match the features
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1c,kp1,img2c,kp2,matches[:20],img1, matchColor= (0,255,255),flags=2)
cv2.imshow('match', img3)
cv2.imwrite('/media/pc/ntfs/Untitled Folder/match.jpg',img3)
cv2.waitKey()
good = matches
# Reshape feature arrays
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# Find the transformation        
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,50.0)
dst = cv2.warpPerspective(img2c, np.linalg.inv(M), (img1.shape[1], img1.shape[0]))
cv2.imshow("Images", dst)
cv2.imwrite('/media/pc/ntfs/Untitled Folder/warped.jpg',dst)
cv2.waitKey()
cv2.destroyAllWindows()
draw_matches(img1c, src_pts, img2c, dst_pts, matches[:15], color=(0,255,255))
cv2.destroyAllWindows()
##    kp2, des2 = sift.detectAndCompute(img2,None)
####    kp2 = star.detect(img1,None)
####    kp2, des2 = brief.compute(img2, kp2)
##
##    matches = bf.knnMatch(des1,des2, k=2)
##
##    good = []
##
##    for m,n in matches:
##        if m.distance < 0.7*n.distance:
##            good.append(m)
##            
##    if len(good)>MIN_MATCH_COUNT:
##        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
##        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
##        
##        M, mask = cv2.findHomography(src_pts, dst_pts) #, cv2.RANSAC,50.0)
##        T = np.dot(M, T)
##        dst = cv2.warpPerspective(img2, np.linalg.inv(T), (img1.shape[1], img1.shape[0]))
##    ##    matchesMask = mask.ravel().tolist()
##    ##    
##    ##    h,w = img1.shape
##    ##    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
##    ##    dst = cv2.perspectiveTransform(pts,M)
##       
##    ##    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
##
##    else:
##        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
####        matchesMask = None
####    plt.figure(1)
####    plt.imshow(img1, cmap='Greys_r')
######    plt.figure(2)
######    plt.imshow(img2, cmap='Greys_r')
####    plt.figure(3)
####    plt.imshow(dst, cmap='Greys_r')
####    plt.show()
##    kp1, des1 = kp2, des2
####    img1 = img2
##    end = time.time()
##    print(end - start)
##    cv2.imshow("Images", dst)
##    t = cv2.waitKey(30) & 0xff
##    if t == 27:
##        break
####draw_params = dict(matchColor = (0,255,0), # draw matches in green color
####                   singlePointColor = None,
####                   matchesMask = matchesMask, # draw only inliers
####                   flags = 2)
####img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
####plt.imshow(img3, 'gray'),plt.show()
##cv2.destroyAllWindows()
