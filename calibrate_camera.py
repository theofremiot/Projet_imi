import cv2 as cv
import numpy as np
import itertools
import matplotlib.pyplot as plt


def calibrate_double_camera(frame1, frame2):
    # créer la matrice avec les coordonnées des points
    pas = 35
    coord_mm = [[i * pas, j * pas, 0] for i, j in itertools.product(range(0, 6), range(0, 8))]
    coord_mm = np.array(coord_mm, dtype=np.float32)

    # trouver les coins des échequiers
    retval1, corners1 = cv.findChessboardCorners(frame1, (8, 6), None)
    retval2, corners2 = cv.findChessboardCorners(frame2, (8, 6), None)

    # dessiner les coins de l'échequier trouvé
    frame1_corners = cv.drawChessboardCorners(frame1, (8, 6), corners1, retval1)
    frame2_corners = cv.drawChessboardCorners(frame2, (8, 6), corners2, retval2)

    # utilisation de la fonction d'opencv pour calibrer ###############################################################
    h1, w1 = frame1.shape[0], frame1.shape[1]
    h2, w2 = frame2.shape[0], frame2.shape[1]

    _, _, dist1, _, _ = cv.calibrateCamera([np.float32(coord_mm)],
                                           [np.float32(corners1)],
                                           (h1, w1),
                                           None,
                                           None)
    _, _, dist2, _, _ = cv.calibrateCamera([np.float32(coord_mm)],
                                           [np.float32(corners2)],
                                           (h2, w2),
                                           None,
                                           None)

    M_int1 = np.zeros((3, 3))
    M_int1[0][0] = 4 * 1280 / 3.58
    M_int1[1][1] = 4 * 960 / 2.02
    M_int1[0][2] = 1280 / 2
    M_int1[1][2] = 960 / 2

    M_int2 = M_int1

    ret1, rvecs1, tvecs1 = cv.solvePnP(np.float32(coord_mm),
                                       corners1,
                                       M_int1,
                                       np.zeros((4, 1)))

    rvecs1, _ = cv.Rodrigues(rvecs1)

    ret2, rvecs2, tvecs2 = cv.solvePnP(np.float32(coord_mm),
                                       corners2,
                                       M_int2,
                                       np.zeros((4, 1)))

    rvecs2, _ = cv.Rodrigues(rvecs2)

    # pour la correspondance aux matrices:
    # https://learnopencv.com/camera-calibration-using-opencv/

    # Obtenir les matrices essentielle et fondamentale ################################################################
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate([np.float32(coord_mm)],
                                                                 [np.float32(corners1)],
                                                                 [np.float32(corners2)],
                                                                 M_int1,
                                                                 dist1,
                                                                 M_int2,
                                                                 dist2,
                                                                 (w1, h1),
                                                                 criteria=criteria,
                                                                 flags=stereocalibration_flags)

    # calculer les points projeter dans le plan image de la caméra 1 ##################################################

    M_ext1 = np.concatenate((rvecs1, tvecs1), axis=1)
    M_ext1 = np.concatenate((M_ext1, [np.array([0, 0, 0, 1])]), axis=0)
    M_int1 = np.concatenate((M_int1, np.array([[0], [0], [0]])), axis=1)
    M_int1 = np.concatenate((M_int1, [np.array([0, 0, 1, 0])]), axis=0)
    M_int1[2, 2] = 1

    M1 = np.dot(M_int1, M_ext1)

    U_list1 = []
    for coord in coord_mm:
        coord_homogenes = np.array([coord[0], coord[1], coord[2], 1])
        alpha = M_ext1[2, 0] * coord[0] + M_ext1[2, 1] * coord[1] + M_ext1[2, 2] * coord[2] + M_ext1[2, 3]
        test = (M_ext1[2, 0], M_ext1[2, 1], M_ext1[2, 2], M_ext1[2, 3])
        U = (1 / alpha) * np.dot(M1, coord_homogenes)
        U_list1.append(U.tolist()[:2])

    img_corners1 = frame1.copy()
    for coor in U_list1:
        coord_tuple = (int(coor[0]), int(coor[1]))
        img_corners1 = cv.circle(img_corners1, coord_tuple, 3, (0, 0, 255), -1)

    # calculer les points projeter dans le plan image de la caméra 2 ##################################################

    M_ext2 = np.concatenate((rvecs2, tvecs2), axis=1)
    M_ext2 = np.concatenate((M_ext2, [np.array([0, 0, 0, 1])]), axis=0)
    M_int2 = np.concatenate((M_int2, np.array([[0], [0], [0]])), axis=1)
    M_int2 = np.concatenate((M_int2, [np.array([0, 0, 1, 0])]), axis=0)
    M_int2[2, 2] = 1

    M2 = np.dot(M_int2, M_ext2)

    U_list2 = []
    for coord in coord_mm:
        coord_homogenes = np.array([coord[0], coord[1], coord[2], 1])
        alpha = M_ext2[2, 0] * coord[0] + M_ext2[2, 1] * coord[1] + M_ext2[2, 2] * coord[2] + M_ext2[2, 3]
        U = (1 / alpha) * np.dot(M2, coord_homogenes)
        U_list2.append(U.tolist()[:2])

    img_corners2 = frame2.copy()
    for coor in U_list2:
        coord_tuple = (int(coor[0]), int(coor[1]))
        img_corners2 = cv.circle(img_corners2, coord_tuple, 3, (0, 0, 255), -1)

    # afficher les points projetés dans le plan image de la caméra sur les deux images ################################

    img_corners = cv.hconcat([img_corners1, img_corners2])
    plt.figure(1)
    plt.imshow(img_corners)

    # afficher les points dans le repère caméra #######################################################################

    coord_cam_x = [coord[0] for coord in coord_mm]
    coord_cam_y = [coord[1] for coord in coord_mm]
    coord_cam_z = [coord[2] for coord in coord_mm]

    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.scatter3D(coord_cam_x, coord_cam_y, coord_cam_z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    M_ext_inv1 = np.linalg.pinv(M_ext1)
    M_ext_inv2 = np.linalg.pinv(M_ext2)
    ax.scatter3D(M_ext_inv1[0, 3], M_ext_inv1[1, 3], M_ext_inv1[2, 3])
    ax.scatter3D(M_ext_inv2[0, 3], M_ext_inv2[1, 3], M_ext_inv2[2, 3])
    plt.show()

    ###################################################################################################################

    return None
