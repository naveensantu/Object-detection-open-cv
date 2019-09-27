import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

def display(img,cmap="gray"):
    fig = plt.figure(figsize=(12,10))
    ax= fig.add_subplot(111)
    ax.imshow(img,cmap="gray")

reeses = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\reeses_puffs.png",0)
display(reeses, cmap="gray")

cereals =cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\many_cereals.jpg", 0)
display(cereals,cmap="gray")
orb = cv2.ORB_create()
kp1,des1=orb.detectAndCompute(reeses,None)
kp2,des2=orb.detectAndCompute(cereals,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)

matches = bf.match(des1,des2)

matches = sorted(matches,key=lambda x:x.distance)
reeses_matches = cv2.drawMatches(reeses,kp1 ,cereals,kp2,matches[:25],None,flags=2)
display(reeses_matches)
