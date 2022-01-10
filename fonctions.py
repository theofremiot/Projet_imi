
import numpy as np
import cv2 as cv
import imutils
import argparse
from collections import deque


def calibrate_camera(ret, corners,coord_mm,h,w):

    coord_px = []
    objpoints=[]
    imgpoints=[]
    mtx=np.zeros((3,3))
    mtx[0][0]=4*1280/3.58
    mtx[1][1]=4*960/2.02
    mtx[0][2]=1280/2
    mtx[1][2]=960/2

    for i in range (corners.shape[0]):
        coord_prev = []
        coord_prev.append(corners[i][0][0])
        coord_prev.append(corners[i][0][1])
        coord_px.append(coord_prev)
    objpoints.append(coord_mm)
    imgpoints.append(coord_px)
    ret, mtx1, dist, rvecs, tvecs = cv.calibrateCamera(np.float32(objpoints), np.float32(imgpoints), (h,w), None, None)
    ret,rvecs,tvecs=cv.solvePnP(np.float32(np.asarray(objpoints[0])),np.asarray(imgpoints[0]),mtx,dist,None,None,False,flags = cv.SOLVEPNP_ITERATIVE )
    
    return(ret,mtx,rvecs,tvecs,objpoints,imgpoints,dist)




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

def ball_tracing(frame):
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())
    greenLower = (10, 120, 180)
    greenUpper = (20, 220, 250)
    ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
    args = vars(ap.parse_args())
    pts = deque(maxlen=args["buffer"])
    blurred = cv.GaussianBlur(frame, (11, 11), 0)

    hsv= cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv.inRange(hsv, greenLower, greenUpper)
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
	# only proceed if at least one contour was found
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv.contourArea)
        
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
        if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
            cv.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
    pts.appendleft(center)
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        print('centre en ', pts[i])
    return frame
