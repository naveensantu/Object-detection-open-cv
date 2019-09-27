import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


img = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\internal_external.png",0)
img.shape
plt.imshow(img,cmap="gray")

image,contour,hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(image.shape)

for i in range(len(contour)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contour, i, 255, -1)

plt.imshow(external_contours,cmap="gray")
internal_contours = np.zeros(image.shape)
for i in range(len(contour)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contour, i, 255, -1)


plt.imshow(internal_contours,cmap="gray")
