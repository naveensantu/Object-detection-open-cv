import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

image= cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\sammy_face.jpg")
plt.imshow(image)
edges= cv2.Canny(image=image, threshold1=100, threshold2=100)
plt.imshow(edges)
med_val = np.median(image)
med_val
lower = int(max(0,0.7*med_val))
upper =int(min(255,1.3*med_val))
edges1= cv2.Canny(image=image, threshold1=lower, threshold2=upper)
plt.imshow(edges1)
blur_image=cv2.blur(image, ksize=(5,5))
edges2= cv2.Canny(image=blur_image, threshold1=lower, threshold2=upper)
plt.imshow(edges2)
