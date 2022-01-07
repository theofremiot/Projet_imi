
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



def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

