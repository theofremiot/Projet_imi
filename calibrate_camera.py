import cv2 as cv
import numpy as np
import itertools


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

    # concatener les deux images
    dst = cv.hconcat([frame1_corners, frame2_corners])
    # afficher l'image concatenée
    cv.imshow('frame', dst)
    cv.waitKey(0)

    # utilisation de la fonction d'opencv pour calibrer
    h1, w1 = frame1.shape[0], frame1.shape[1]
    h2, w2 = frame2.shape[0], frame2.shape[1]
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera([np.float32(coord_mm)],
                                                           [np.float32(corners1)],
                                                           (h1, w1),
                                                           None,
                                                           None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera([np.float32(coord_mm)],
                                                           [np.float32(corners2)],
                                                           (h2, w2),
                                                           None,
                                                           None)



    return None
