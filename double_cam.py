import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fonctions import calibrate_camera,calibration_stereo
from fonctions import DLT,ball_tracing
from mpl_toolkits.mplot3d import Axes3D

cap0 = cv.VideoCapture(1)
cap1 = cv.VideoCapture(2)
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



positions=[]
res=None
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0,pts0= ball_tracing(frame0)
    frame1,pts1= ball_tracing(frame1)
    print(pts0[0])
    if(pts0[0]!=None and pts1[0]!=None):
        res=DLT(M0,M1,np.asarray(pts0[0]),np.asarray(pts1[0]))   
    if(np.shape(res)!=0):
        positions.append(res)
    dst = cv.hconcat([frame0,frame1]) 
    cv.imshow('frame',dst)
    if cv.waitKey(1) == ord('q'):
        break
    
print(positions)
cap0.release()
cap1.release()