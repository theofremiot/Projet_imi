
import numpy as np
import cv2 as cv


def calibrate_camera(ret, corners,coord_mm,h,w):
    coord_px = []
    objpoints=[]
    imgpoints=[]


    for i in range (corners.shape[0]):
        coord_prev = []
        coord_prev.append(corners[i][0][0])
        coord_prev.append(corners[i][0][1])
        coord_px.append(coord_prev)
    objpoints.append(coord_mm)
    imgpoints.append(coord_px)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(np.float32(objpoints), np.float32(imgpoints), (h,w), None, None)
    return(ret,mtx,dist,rvecs,tvecs,objpoints,imgpoints)





