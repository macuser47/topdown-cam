import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 7
HEIGHT = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cam = cv.VideoCapture(0)

while True:
    _, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners

    ret, corners = cv.findChessboardCorners(gray, (WIDTH,HEIGHT), None)
    # If found, add object points, image points (after refining them)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            print("Calibration matrix: {}".format(mtx))
            print("Dist: {}".format(dist))

        cv.waitKey(500)

cv.destroyAllWindows()
