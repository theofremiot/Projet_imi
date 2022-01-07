import numpy as np
import cv2 as cv


cap0 = cv.VideoCapture(1)
cap1 = cv.VideoCapture(2)
ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

if not cap1.isOpened():
    print("Cannot open camera0")
    exit()
if not cap1.isOpened():
    print("Cannot open camera1")
    exit()
count = 0


trajectory = []

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


    gray = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

    gray_blurred = cv.blur(gray, (3, 3)) 
        
    detected_circles = cv.HoughCircles(gray_blurred,  
                        cv.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                    param2 = 30, minRadius = 32, maxRadius = 60)
        
    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles)) 
    
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
    
            
            cv.circle(frame0, (a, b), r, (0, 255, 0), 2) 
            print('a = ', a)
            print('b = ', b)
            print('r = ', r)
            
            cv.circle(frame0, (a, b), 1, (0, 0, 255), 3)
            print('coord centre balle = ', b, a)
            trajectory.append([b,a])
            #cv.imshow("Detected Circle", frame0) 
            if(np.shape(trajectory)!=[0,0]):
                for pos in trajectory:
                    frame0[pos[0],pos[1]]=[255,0,0]

    
    dst = cv.hconcat([frame0,frame1]) 
    # Display the resulting frame
    
    cv.imshow('frame', dst)

    if cv.waitKey(1) == ord('a'):
        break
