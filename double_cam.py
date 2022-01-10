import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fonctions import calibrate_camera
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
        test=np.copy(frame0)
        cv.drawChessboardCorners(frame0, (8,6), corners0, ret0)
        cv.drawChessboardCorners(frame1, (8,6), corners1, ret1)

    dst = cv.hconcat([frame0,frame1]) 
    cv.imshow('frame',dst)
    if cv.waitKey(1) == ord('a'):
        break
cap0.release()
cap1.release()

#cv.solvePnP()
#ajout parametre intrinseques 

# taille capteur = 3.58*2.02 mm^2
# fx=focale_mm*dim_image(px)/dim_capteur(mm)
# cx,cy =definition en px/2



ret0, mtx0,  rvecs0, tvecs0, objpoints0, imgpoints0, dist0= calibrate_camera(ret0,corners0,coord_mm,h,w)

###Matrice passage de camera 0###
R0=cv.Rodrigues(np.float32(rvecs0))
# print('rbis = ', Rbis)
# print('size rbis',Rbis[0].shape)
# T=np.concatenate(Rbis,tvecs)
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
# print('rbis = ', Rbis)
# print('size rbis',Rbis[0].shape)
# T=np.concatenate(Rbis,tvecs)

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

coord_proj=np.zeros((49,3))
for i in range(48):
    coord_hom=[float(coord_mm[i][0]),float(coord_mm[i][1]),float(coord_mm[i][2]),1]
    coord_proj[i]=np.dot(P1,(coord_hom))


tvecs1= RT @  np.transpose(tvecs1) 
tvecs0= RT @  np.transpose(tvecs0)
ax.scatter3D(tvecs0[0],tvecs0[1],tvecs0[2])
ax.scatter3D(tvecs1[0],tvecs1[1],tvecs1[2])
plt.show()


for i in range(np.shape(coord_mm)[0]):
    alpha=M1[2][0]*coord_mm[i][0]+M1[2][1]*coord_mm[i][1]+M1[2][2]*coord_mm[i][2]+M1[2][3]
    image = cv.circle(test, (int(coord_proj[i][0]/alpha),int(coord_proj[i][1]/alpha)), radius=5, color=(0, 0, 255), thickness=-1)

cv.imshow('res',image)
cv.waitKey(0)





cv.destroyAllWindows()