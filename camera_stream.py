import cv2 as cv

cap0 = cv.VideoCapture(0)
cap1 = cv.VideoCapture(2)
ret0, frame0 = cap0.read()

ret1, frame1 = cap1.read()

if not cap1.isOpened():
    print("Cannot open camera0")
    exit()
if not cap1.isOpened():
    print("Cannot open camera1")
    exit()
while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    ret0, frame0 = cap0.read()
    # if frame is read correctly ret is True
    if not ret1:
        print("Can't receive frame 1 (stream end?). Exiting ...")
        break
    if not ret0:
        print("Can't receive frame 0 (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    # gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    retval1, corners1 = cv.findChessboardCorners(frame1, (8, 6), None)
    retval0, corners0 = cv.findChessboardCorners(frame0, (8, 6), None)

    cv.drawChessboardCorners(frame1, (8, 6), corners1, retval1)
    cv.drawChessboardCorners(frame0, (8, 6), corners0, retval0)

    dst = cv.hconcat([frame0, frame1])
    # Display the resulting frame
    cv.imshow('frame', dst)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap0.release()
cap1.release()
cv.destroyAllWindows()
