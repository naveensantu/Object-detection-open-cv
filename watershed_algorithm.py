import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def display(img,cmap="gray"):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap="gray")

## segment detection without water shed:
sep_coins = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\pennies.jpg")
display(sep_coins)
sep_blur = cv2.medianBlur(sep_coins, 25)
display(sep_blur)
sep_gray = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)
display(sep_gray)
ret , sep_thresh = cv2.threshold(sep_gray, 160, 255, cv2.THRESH_BINARY_INV)
display(sep_thresh)
image,contours,hierarchy = cv2.findContours(sep_thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for i in range (len(contours)):
    if hierarchy[0][i][3]==-1:
        cv2.drawContours(sep_coins,contours, i, (255,0,0),10)
display(sep_coins)

## water shed
image_coins = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\pennies.jpg")
image_coins = cv2.medianBlur(image_coins, 35)
gray_coins = cv2.cvtColor(image_coins, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
display(opening)
back_grnd = cv2.dilate(opening,kernel,iterations=3)
display(back_grnd)

dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, thresh_dt = cv2.threshold(dist_trans, 0.7* dist_trans.max(), 255,0)
display(thresh_dt)
thresh_dt=np.uint8(thresh_dt)
unknown = cv2.subtract(back_grnd ,thresh_dt)
display(unknown)

ret,markers = cv2.connectedComponents(thresh_dt)
markers = markers +1

markers[unknown==255]=0

display(markers)
markers = cv2.watershed(image_coins,markers)
display(markers)
image,contours,hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for i in range (len(contours)):
    if hierarchy[0][i][3]==-1:
        cv2.drawContours(sep_coins,contours, i, (255,0,0),10)

display(sep_coins)
