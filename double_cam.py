import numpy as np
import cv2 as cv
from fonctions import calibrate_camera


cap0 = cv.VideoCapture(1)
cap1 = cv.VideoCapture(2)

ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

h=frame0.shape[1]
w=frame0.shape[0]


if not cap0.isOpened():
    print("Cannot open camera0")
    exit()
if not cap1.isOpened():
    print("Cannot open camera1")
    exit()



coord_mm=[]
coord_mm2=[]
coord_px=[]
coord_px2=[]
pas=35

for i in range(0,6):
    for j in range(0,8):
        coord_mm.append([j*pas,i*pas,0])



for i in range(0,6):
    for j in range(0,8):
        coord_mm2.append([(7-j)*pas,(6-i)*pas,0])



while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret0, corners0 = cv.findChessboardCorners(frame0, (8,6), None)
    ret1, corners1 = cv.findChessboardCorners(frame1, (8,6), None)
    cv.drawChessboardCorners(frame0, (8,6), corners0, ret0)
    cv.drawChessboardCorners(frame1, (8,6), corners1, ret1)
    dst = cv.hconcat([frame0,frame1]) 
    cv.imshow('frame',dst)
    if cv.waitKey(1) == ord('a'):
        break
cap0.release()
cap1.release()

ret0, mtx0, dist0, rvecs0, tvecs0, objpoints0, imgpoints0= calibrate_camera(ret0,corners0,coord_mm,h,w)
ret1, mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1= calibrate_camera(ret1,corners1,coord_mm2,h,w)

print('ret = ', ret1)
print('mtx = ', mtx1)
print('dist = ', dist1)
print('rvecs = ', rvecs1)
print('tvecs = ', tvecs1)
print('size tvecs = ', tvecs1[0].shape)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(np.float32(objpoints0), np.float32(imgpoints0), np.float32(imgpoints1), mtx0, dist0,
mtx1, dist1, (w,h), criteria = criteria, flags = stereocalibration_flags)

print('rot=', R)
print('T=', T)



    


# When everything done, release the capture


cv.destroyAllWindows()