import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

flat_chess=cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\flat_chessboard.png")
plt.imshow(flat_chess)
found,corners = cv2.findChessboardCorners(flat_chess, (7,7))
found

draw=cv2.drawChessboardCorners(flat_chess, (7,7), corners, found)
plt.imshow(draw)

dots = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\dot_grid.png")
found,corners = cv2.findCirclesGrid(dots,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)
draw1=cv2.drawChessboardCorners(dots, (10,10), corners, found)
plt.imshow(draw1)
