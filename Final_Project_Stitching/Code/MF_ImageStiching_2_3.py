# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 08:30:16 2021

@author: genti
Enviromnment CNN2
"""
#%%========== Loading libraries ================
import sys
sys.path.append('/usr/local/lib/python3.9/site-packages') # replace with the path where cv2 is installed on your system

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import imutils
cv2.ocl.setUseOpenCL(False)
import tkinter as tk
from tkinter import filedialog
#%%========== select the image id (valid values 1,2,3, or 4)
feature_extractor = 'sift' # one of 'sift', 'surf', 'brisk', 'orb' #it was orb
feature_matching = 'knn' #it was bf
#%%============== Directory choosing
#root = tk.Tk()
#root.withdraw()
#main_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick a directory for the project')
#main_dir= 'D:/Google Drive/Collaborations/01_Current/Digital_Ag/Digital_Ag_Class/01_General/Notes/Image_Processing/Image_Stitching'
#%%========== read first image and transform it to grayscale
# Make sure that the train image is the image that will be transformed
trainImg = imageio.imread('../Data/Raw/L1.jpg')
#trainImg = imageio.imread(main_dir+'/Data/Raw/A1.jpg')

trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
#%%========== read second images and transform it to grayscale
queryImg = imageio.imread('../Data/Raw/L2.jpg')
# Opencv defines the color channel in the order BGR. 
# Transform it to RGB to be compatible to matplotlib
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY) #change color, to gray because of rgb2gray command
#%%========== plot images side by side ===========================
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
ax1.imshow(queryImg, cmap="gray")
ax1.set_xlabel("Left Image (Static)", fontsize=14) #template, it is not going to change (static)

ax2.imshow(trainImg, cmap="gray")
ax2.set_xlabel("Right Image (Image to be transformed)", fontsize=14)

plt.show()
#%%========== declearing a feature selection function =========== 
def detectAndDescribe(image, method=None): #The function detectAndDescribe(image, method=None) takes an image and a feature detection method as input and computes key points and feature descriptors using the specified method.
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)
			
#%%========== Finding features in the images ==================
#The code uses the detectAndDescribe() function to extract key points (kpsA, kpsB) and feature descriptors (featuresA, featuresB) from the grayscale images trainImg_gray and queryImg_gray using the specified feature_extractor method.
kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)
#%%========== Display the keypoints and features detected on both images ========
#The code creates a figure with two subplots (ax1 and ax2) and displays the images trainImg_gray and queryImg_gray with the detected key points (kpsA and kpsB) drawn on them using the cv2.drawKeypoints() function. The subplots are labeled as "(a)" and "(b)" respectively, and the figure is displayed using plt.show().
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(trainImg_gray,kpsA,None,color=(0,255,0)))
ax1.set_xlabel("(a)", fontsize=14)
ax2.imshow(cv2.drawKeypoints(queryImg_gray,kpsB,None,color=(0,255,0)))
ax2.set_xlabel("(b)", fontsize=14)

plt.show()
#%%========= Create a matching function =========================================
def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf
#%%======== Create a matching function that uses brute force matching ==========
def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches
#%%======== Create a matching function that uses k nearest neighbors matching ==========		
def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches			
			
#%%======== Applying a matching algorithm and showing matched pairs.=====================
print("Using: {} feature matcher".format(feature_matching))

fig = plt.figure(figsize=(20,8))

if feature_matching == 'bf':
    matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100],
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
elif feature_matching == 'knn':
    matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,np.random.choice(matches,100),
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

plt.imshow(img3)
plt.show()			

#%%====== Define a homography, i.e. a way to shift the images to align =================

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None
#%%===== Compute the homography =========================
M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
if M is None:
    print("Error!")
(matches, H, status) = M
print(H)

#%%==== Create the stiching (panorama)==================
width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]

result = cv2.warpPerspective(trainImg, H, (width, height))
result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

plt.figure(figsize=(20,10))
plt.imshow(result)

plt.axis('off')
plt.show()
#%%==== Transform the panorama image to grayscale, show it and and threshold it =======
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(20,10))
plt.imshow(gray)

plt.axis('off')
plt.show()
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
#%%==== Finds contours from the binary images=====================
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#%%==== get the maximum contour area =============================
c = max(cnts, key=cv2.contourArea)
# get a bbox from the contour area
(x, y, w, h) = cv2.boundingRect(c)
# crop the image to the bbox coordinates
result = result[y:y + h, x:x + w]
#%%==== Plot the final image =====================================
plt.figure(figsize=(20,10))
plt.imshow(result)

