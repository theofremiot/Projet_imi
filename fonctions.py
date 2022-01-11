
import numpy as np
import cv2 as cv
import imutils
import argparse
from collections import deque
import matplotlib.pyplot as plt


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
    #ret, mtx1, dist, rvecs, tvecs = cv.calibrateCamera(np.float32(objpoints), np.float32(imgpoints), (h,w), None, None)
    dist=np.zeros((4,1))
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
        if( M["m00"]!=0):
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
    return frame,pts

def calibration_stereo(cap0,cap1,h,w):

    coord_mm=[]
    pas=35
    for i in range(0,6):
        for j in range(0,8):
            coord_mm.append([i*pas,j*pas,0])

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret0, corners0 = cv.findChessboardCorners(frame0, (8,6), None)
        ret1, corners1 = cv.findChessboardCorners(frame1, (8,6), None)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if(ret0 == True & ret1==True):
            test0=np.copy(frame0)
            test1=np.copy(frame1)
            cv.drawChessboardCorners(frame0, (8,6), corners0, ret0)
            cv.drawChessboardCorners(frame1, (8,6), corners1, ret1)

        dst = cv.hconcat([frame0,frame1]) 
        cv.imshow('frame',dst)
        if cv.waitKey(1) == ord('a'):
            break
    # cap0.release()
    # cap1.release()

    #cv.solvePnP()
    #ajout parametre intrinseques 

    # taille capteur = 3.58*2.02 mm^2
    # fx=focale_mm*dim_image(px)/dim_capteur(mm)
    # cx,cy =definition en px/2

    ret0, mtx0,  rvecs0, tvecs0, objpoints0, imgpoints0, dist0= calibrate_camera(ret0,corners0,coord_mm,h,w)

    ###Matrice passage de camera 0###
    R0=cv.Rodrigues(np.float32(rvecs0))
    tvecs0=np.transpose(tvecs0)
    M0=np.zeros((4,4))  #Mext
    for i in range(3):
        for j in range(3):
            M0[i][j]=R0[0][i][j]
    M0[0][3]=tvecs0[0][0]
    M0[1][3]=tvecs0[0][1]
    M0[2][3]=tvecs0[0][2]
    M0[3][3]=1
    Mtx0=np.zeros((3,4)) #Mint
    for i in range(3):
        for j in range(3):
            Mtx0[i][j]=mtx0[i][j]
    P0=np.dot(Mtx0,M0)

    ret1, mtx1, rvecs1, tvecs1, objpoints1, imgpoints1, dist1= calibrate_camera(ret1,corners1,coord_mm,h,w)

    ###Matrice passage de camera 1###
    R1=cv.Rodrigues(np.float32(rvecs1))
    tvecs1=np.transpose(tvecs1)
    M1=np.zeros((4,4))  #Mext
    for i in range(3):
        for j in range(3):
            M1[i][j]=R1[0][i][j]
    M1[0][3]=tvecs1[0][0]
    M1[1][3]=tvecs1[0][1]
    M1[2][3]=tvecs1[0][2]
    M1[3][3]=1
    Mtx1=np.zeros((3,4)) #Mint
    for i in range(3):
        for j in range(3):
            Mtx1[i][j]=mtx1[i][j]
    P1=np.dot(Mtx1,M1)



    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(np.float32(objpoints0), np.float32(imgpoints0), np.float32(imgpoints1), mtx0, dist0,
    mtx1, dist1, (w,h), criteria = criteria, flags = stereocalibration_flags)

    [dst,jacobian] = cv.Rodrigues(np.float32(R))
    dst=dst*180/np.pi

    print('rot=', dst)
    print('T=', T)



    RT = np.concatenate([R, np.transpose(T)])
    projections =[]
    P=np.dot(Mtx0,RT)

    for i in range(np.shape(coord_mm)[0]):
        coord_hom=[float(coord_mm[i][0]),float(coord_mm[i][1]),float(coord_mm[i][2]),1]
        projections.append(coord_hom)
    coord_cam_x = [coord[0] for coord in projections]
    coord_cam_y = [coord[1] for coord in projections]
    coord_cam_z = [coord[2] for coord in projections]

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(coord_cam_x, coord_cam_y, coord_cam_z)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')


    M0_inv= np.linalg.pinv(M0) 
    M1_inv= np.linalg.pinv(M1) 
    ax.scatter3D(M0_inv[0][3],M0_inv[1][3],M0_inv[2][3])
    ax.scatter3D(M1_inv[0][3],M1_inv[1][3],M1_inv[2][3])
    plt.show()

    coord_proj0=np.zeros((48,3))
    coord_proj1=np.zeros((48,3))
    for i in range(48):
        coord_hom=[float(coord_mm[i][0]),float(coord_mm[i][1]),float(coord_mm[i][2]),1]
        coord_proj0[i]=np.dot(P0,(coord_hom))
    for i in range(48):
        coord_hom=[float(coord_mm[i][0]),float(coord_mm[i][1]),float(coord_mm[i][2]),1]
        coord_proj1[i]=np.dot(P1,(coord_hom))


    for i in range(np.shape(coord_mm)[0]):
        alpha=M0[2][0]*coord_mm[i][0]+M0[2][1]*coord_mm[i][1]+M0[2][2]*coord_mm[i][2]+M0[2][3]
        image0 = cv.circle(test0, (int(coord_proj0[i][0]/alpha),int(coord_proj0[i][1]/alpha)), radius=5, color=(0, 0, 255), thickness=-1)
    for i in range(np.shape(coord_mm)[0]):
        alpha=M1[2][0]*coord_mm[i][0]+M1[2][1]*coord_mm[i][1]+M1[2][2]*coord_mm[i][2]+M1[2][3]
        image1 = cv.circle(test1, (int(coord_proj1[i][0]/alpha),int(coord_proj1[i][1]/alpha)), radius=5, color=(0, 0, 255), thickness=-1)


    dst = cv.hconcat([image0,image1]) 
    cv.imshow('res',dst)
    cv.waitKey(0)

    cv.destroyAllWindows()
    return M0,M1,P0,P1