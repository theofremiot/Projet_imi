import ball_detection
import calibrate_camera
import cv2 as cv

if __name__ == "__main__":

    print("STARTING...")

    frame1 = cv.imread("data/images0/frame0.jpg")
    frame2 = cv.imread("data/images1/frame0.jpg")

    calibrate_camera.calibrate_double_camera(frame1, frame2)

    print("PROGRAM STOPPED")
