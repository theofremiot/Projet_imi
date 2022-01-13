import cv2  as cv
  
   
cap0 = cv.VideoCapture(1)
cap1 = cv.VideoCapture(2)
   
if (cap0.isOpened() == False):  
    print("Error reading video file") 
  
frame_width = int(cap0.get(3)) 
frame_height = int(cap0.get(4)) 
   
size = (frame_width, frame_height) 
   
result1 = cv.VideoWriter('/Users/Matthieu/Desktop/video/cam1.avi',  
                         cv.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
result2 = cv.VideoWriter('/Users/Matthieu/Desktop/video/cam2.avi',  
                         cv.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
    
    
while(True): 
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if ret1 == True:  
        dst = cv.hconcat([frame0,frame1]) 
        cv.imshow('frame',dst)

        result1.write(frame0) 
        result2.write(frame1) 
        if cv.waitKey(1) & 0xFF == ord('s'): 
            break
  
    
    else: 
        break
  
cap0.release()
cap1.release()

result1.release()
result2.release()

cv.destroyAllWindows() 
   
print("The video was successfully saved") 