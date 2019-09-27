import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


flat_chess=cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\flat_chessboard.png")
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess,cmap="gray")
real_chess=cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\real_chessboard.jpg")
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)
gray_real_chess =cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess,cmap='gray')
gray = np.float32(gray_flat_chess)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst,None)
flat_chess[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(flat_chess)

gray_real = np.float32(gray_real_chess)
dst1 = cv2.cornerHarris(src=gray_real, blockSize=2, ksize=3, k=0.04)
dst1 = cv2.dilate(dst1,None)
real_chess[dst1>0.01*dst1.max()] = [255,0,0]
plt.imshow(real_chess)
###RELOADING THE SAME IMAGES
flat_chess2=cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\flat_chessboard.png")
flat_chess2= cv2.cvtColor(flat_chess2, cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess2)
gray_flat_chess2 = cv2.cvtColor(flat_chess2,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess2,cmap="gray")
real_chess2=cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\real_chessboard.jpg")
real_chess2 = cv2.cvtColor(real_chess2, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess2)
gray_real_chess2 =cv2.cvtColor(real_chess2,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess2,cmap='gray')

corners = cv2.goodFeaturesToTrack(gray_flat_chess2, 5, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(flat_chess2,(x,y),3,(255,0,0),-1)
plt.imshow(flat_chess2)
corners1 = cv2.goodFeaturesToTrack(gray_real_chess2, 75, 0.01, 10)
corners1 = np.int0(corners1)
for i in corners1:
    x,y = i.ravel()
    cv2.circle(real_chess2,(x,y),3,(255,0,0),-1)
plt.imshow(real_chess2)
