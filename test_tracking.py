import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from fonctions import ball_tracing, calibrate_camera
from fonctions import DLT


from mpl_toolkits.mplot3d import Axes3D

cap0 = cv.VideoCapture(1)
cap1 = cv.VideoCapture(2)


# caP0.set(cv.CAP_PROP_FPS, 60)
# cap0.set(cv.CAP_PROP_FPS, 60)

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





while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0= ball_tracing(frame0)
    frame1= ball_tracing(frame1)




    dst = cv.hconcat([frame0,frame1]) 
    cv.imshow('frame',dst)

    if cv.waitKey(1) == ord('a'):
        break
    
cap0.release()
cap1.release()



# ret0, mtx0, dist0, rvecs0, tvecs0, objpoints0, imgpoints0= calibrate_camera(ret0,corners0,coord_mm,h,w)
# ret1, mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1= calibrate_camera(ret1,corners1,coord_mm,h,w)


# print('ret = ', ret1)
# print('mtx = ', mtx1)
# print('dist = ', dist1)
# print('rvecs = ', rvecs1)
# print('tvecs = ', tvecs1)
# print('size tvecs = ', tvecs1[0].shape)


# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
# stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
# ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(np.float32(objpoints0), np.float32(imgpoints0), np.float32(imgpoints1), mtx0, dist0,
# mtx1, dist1, (w,h), criteria = criteria, flags = stereocalibration_flags)

# [dst,jacobian] = cv.Rodrigues(np.float32(R))
# dst=dst*180/np.pi

# print('rot=', dst)
# print('T=', T)

# #RT matrix for C1 is identity.
# RT0 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
# P0 = mtx0 @ RT0 #projection matrix for C1
 
# #RT matrix for C2 is the R and T obtained from stereo calibration.
# RT1 = np.concatenate([R, T], axis = -1)
# P1 = mtx1 @ RT1 #projection matrix for C2



# coord_proj=np.zeros_like(coord_mm)


# for i in range(np.shape(coord_mm)[0]):
#     coord_hom=[float(coord_mm[i][0]),float(coord_mm[i][1]),float(coord_mm[i][2]),1]
#     coord_proj[i]=np.dot(P0,(coord_hom))

# coord_cam_x = [coord[0] for coord in coord_proj]
# coord_cam_y = [coord[1] for coord in coord_proj]
# coord_cam_z = [coord[2] for coord in coord_proj]

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(coord_cam_x, coord_cam_y, coord_cam_z)
# ax.set_xlabel('z')
# ax.set_ylabel('x')
# ax.set_zlabel('y')

# ax.scatter3D(T[0],T[1],T[2])
# plt.show()




cv.destroyAllWindows()