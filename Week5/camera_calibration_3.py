import cv2
import numpy as np
import glob

# Chessboard dimensions
CHECKERBOARD = (6, 9)

# Prepare object points like (0,0,0), (1,0,0), ...,(5,8,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []  # 3d point in real world
imgpoints = []  # 2d points in image plane

# You could add more images here using glob
images = [r'C:\ComputerVision-main\Week5\6ca9d13083d20d8c54c3.jpg']  # or: images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (400,400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        img_disp = img.copy()
        # draw corners for display
        cv2.drawChessboardCorners(img_disp, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img_disp)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibration: need at least one good set of corners
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Undistort the image
    img = cv2.imread(images[0])
    img = cv2.resize(img, (400,400))
    h, w = img.shape[:2]
    newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcam_mtx)

    cv2.imwrite('undistorted_result.jpg', undistorted_img)
    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Checkerboard couldn't be found. Please use a clearer image or more images for calibration.")