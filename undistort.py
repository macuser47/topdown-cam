import numpy as np
import cv2 as cv

mtx = np.array( 
    [[465.12139372,   0.,         346.19911664],
     [  0.,         482.94398612, 221.13345464],
     [  0.,           0.,           1.,        ]])
dist = np.array([[-8.04944916e-02,  9.73581185e-02, -8.29301732e-04,  9.82650195e-05, -7.80368225e-02]])

cam = cv.VideoCapture(0)
_, img = cam.read()
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

pts = np.array([
    [286, 239], [333, 239], [381, 240], [430, 241],
    [232, 336], [324, 341], [413, 343], [508, 345]
])
pts_mapped = np.array([
    [160+640/2, 144], [160+640/2, 144+128/2], [160+640/2, 144+256/2], [160+640/2, 144+384/2],
    [160, 144], [160, 144+128/2], [160, 144+256/2], [160, 144+384/2]
])
H, _ = cv.findHomography(pts, pts_mapped)
print("H: {}".format(H))

while True:
    _, img = cam.read()
    img_undistort = cv.undistort(img, mtx, dist, None, newcameramtx)
    img_pshift = cv.warpPerspective(img_undistort, H, (img.shape[1], img.shape[0]))
    cv.imshow(
        'img',
        np.concatenate((img_undistort, img_pshift), axis=1)
    )
    cv.waitKey(500)
