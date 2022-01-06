import numpy as np
import cv2 as cv


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
        coord_mm2.append([(7-j)*pas,(6-i)*pas,150])



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

for i in range (corners0.shape[0]):
    coord_prev = []
    coord_prev.append(corners0[i][0][0])
    coord_prev.append(corners0[i][0][1])
    coord_px.append(coord_prev)
coord_prev=[]
for i in range (corners1.shape[0]):
    coord_prev = []
    coord_prev.append(corners1[i][0][0])
    coord_prev.append(corners1[i][0][1])
    coord_px2.append(coord_prev)




objpoints = [] 
imgpoints = []
objpoints.append(coord_mm)
objpoints.append(coord_mm2)
imgpoints.append(coord_px)
imgpoints.append(coord_px2)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(np.float32(objpoints), np.float32(imgpoints), (h,w), None, None)

print('mtx',mtx)


    


# When everything done, release the capture
cap0.release()

cv.destroyAllWindows()