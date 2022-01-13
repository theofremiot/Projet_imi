from asyncio import sleep
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fonctions import calibrate_camera,calibration_stereo
from fonctions import DLT,ball_tracing,create_csv_file, calcul_vitesse
from mpl_toolkits.mplot3d import Axes3D
import time



cap0 = cv.VideoCapture("/Users/Matthieu/Desktop/video/cam1.avi")
cap1 = cv.VideoCapture("/Users/Matthieu/Desktop/video/cam2.avi")
ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

w=frame0.shape[1]
h=frame0.shape[0]


if not cap0.isOpened():
    print("Cannot open camera0")
    exit()
if not cap1.isOpened():
    print("Cannot open camera1")
    exit()

M0,M1,P0,P1=calibration_stereo(cap0,cap1,h,w)
cap0.release()
cap1.release()



cap0 = cv.VideoCapture("/Users/Matthieu/Desktop/video/cam1.avi")
cap1 = cv.VideoCapture("/Users/Matthieu/Desktop/video/cam2.avi")


# print(P0)
# print(np.linalg.pinv(P0))

positions=[]
res=None

start=time.time()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if(ret1==False):
        break
    frame0,pts0= ball_tracing(frame0)
    frame1,pts1= ball_tracing(frame1)
    #print(pts0[0])
    if(pts0[0]!=None and pts1[0]!=None):
        res=DLT(P0,P1,np.asarray(pts0[0]),np.asarray(pts1[0])) 
        stop=time.time()
        res=[res[0],res[1],res[2],stop-start]
    if(np.shape(res)!=0):
        stop=time.time()
        positions.append(res)
    dst = cv.hconcat([frame0,frame1]) 
    cv.imshow('frame',dst)
    if cv.waitKey(1) == ord('q'):
        break
    

positions_filtered = []
for pos in positions:
    if(pos is not None):
        positions_filtered.append(pos)


coord_cam_x = [coord[0] for coord in positions_filtered]
coord_cam_y = [coord[1] for coord in positions_filtered]
coord_cam_z = [coord[2] for coord in positions_filtered]

plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')

M0_inv= np.linalg.pinv(M0) 
M1_inv= np.linalg.pinv(M1) 
ax.scatter3D(M0_inv[0][3],M0_inv[1][3],M0_inv[2][3])
ax.scatter3D(M1_inv[0][3],M1_inv[1][3],M1_inv[2][3])

for i in range(len(coord_cam_x)):
    ax.scatter3D(coord_cam_x[i], coord_cam_y[i], coord_cam_z[i],c='blue')
    plt.pause(0.2)


plt.show()
plt.figure()



x=[]
y=[]
for pos in positions:
    if(pos is not None):
        x.append(pos[0])
        y.append(pos[1])
plt.plot(x,y)
plt.show()

vit=calcul_vitesse(positions_filtered,5)
print(vit) #mm/s
create_csv_file(positions_filtered)