import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

full_image = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\sammy.jpg")
full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
plt.imshow(full_image)


face_image = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\sammy_face.jpg")
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
plt.imshow(face_image)
full_image.shape
face_image.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for i in methods:
    full_image_c1 = full_image.copy()
    method = eval(i)
    res = cv2.matchTemplate(full_image_c1, face_image, method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height,width,channels=face_image.shape
    bottom_right = (top_left[0]+width,top_left[1]+height)

    cv2.rectangle(full_image_c1,top_left,bottom_right,(255,0,0),10)

    plt.subplot(121)
    plt.imshow(res)
    plt.title ("HEatmap of template matching")

    plt.subplot(122)
    plt.imshow(full_image_c1)
    plt.title("detection of template")
    plt.suptitle(i)

    plt.show()
    print('\n')
    print('\n')
    print('\n')
