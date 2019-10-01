import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

road = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\road_image.jpg")
road_copy = np.copy(road)
plt.imshow(road)

road.shape[:2]
marker_image=np.zeros(road.shape[:2],dtype=np.int32)
segments = np.zeros(road.shape,dtype=np.uint8)

from matplotlib import cm
cm.tab10(0)

def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)

colors=[]
for i in range(10):
    colors.append(create_rgb(i))

colors


#global variables
n_markers = 10
current_marker = 1
marks_updated = False

#callback function
def mouse_call(event,x,y,flags,params):
    global marks_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(marker_image, (x,y), 10, (current_marker),-1)
        cv2.circle(road_copy,(x,y),10,colors[current_marker],-1)
        marks_updated=True

#while true
cv2.namedWindow("road image")
cv2.setMouseCallback("road image",mouse_call)
while True:
    cv2.imshow("watershed segments",segments)
    cv2.imshow("road image",road_copy)

    k=cv2.waitKey(delay=1)
    if k ==27:
        break

    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2],dtype=np.int32)
        segments = np.zeros(road.shape,dtype=np.uint8)

    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road,marker_image_copy)
        segments = np.zeros(road.shape,dtype=np.uint8)

        for color_ind in range (n_markers):
            segments[marker_image_copy==(color_ind)]=colors[color_ind]

cv2.destroyAllWindows()
