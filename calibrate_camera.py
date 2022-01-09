import cv2 as cv


def calibrate_double_camera(frame1, frame2):

    # TODO: faire la matrice avec les corrdonnées des points

    # trouver les coins des échequiers
    retval1, corners1 = cv.findChessboardCorners(frame1, (8, 6), None)
    retval0, corners0 = cv.findChessboardCorners(frame2, (8, 6), None)

    # dessiner les coins de l'échequier trouvé
    cv.drawChessboardCorners(frame1, (8, 6), corners1, retval1)
    cv.drawChessboardCorners(frame2, (8, 6), corners0, retval0)

    # TODO: utilser la fonction d'opencv pour calibrer

    # concatener les deux images
    dst = cv.hconcat([frame1, frame2])
    # afficher l'image concatenée
    cv.imshow('frame', dst)
    cv.waitKey(0)

    return None
