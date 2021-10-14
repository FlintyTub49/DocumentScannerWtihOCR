# ---------------------------- Importing Packages ---------------------------- #
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import polygon_interacter as poly_i
from DocScanner import DocScanner

import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd

import argparse
import os


# --------------------- Images Stacking Utility Function --------------------- #
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# ------------------------- Main Running Of The Code ------------------------- #
# Setting a fixed resolution of the image
widthImg = 1200
heightImg = 675

cap = cv2.VideoCapture(1)
cap.set(10,150)

interactive = int(input('Do you want it to be interactive? (1/0): '))
docCap = DocScanner(interactive = interactive)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
                     
    # Finding Contours
    screenCnt = docCap.get_contour(img)

    # If the interactive feature has to be used,
    if docCap.interactive:
        # ---------------------- Experimental - For Ease Of Use ---------------------- #
        screenCnt[0][0] = screenCnt[0][0] - 10; screenCnt[0][1] = screenCnt[0][1] + 10; 
        screenCnt[1][0] = screenCnt[1][0] - 10; screenCnt[1][1] = screenCnt[1][1] - 10; 
        screenCnt[2][0] = screenCnt[2][0] + 10; screenCnt[2][1] = screenCnt[2][1] - 10; 
        screenCnt[3][0] = screenCnt[3][0] + 10; screenCnt[3][1] = screenCnt[3][1] + 10; 
        # ---------------------------------------------------------------------------- #
        screenCnt = docCap.interactive_get_contour(screenCnt, img)

    # Applying the perspective transformation
    warped = docCap.four_point_transform(img, screenCnt)

    # Preprocessing image for OCR
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sharpen1 = cv2.GaussianBlur(gray, (0,0), 3)
    sharpen2 = cv2.addWeighted(gray, 1.5, sharpen1, -0.5, 0)
    
    imageArray = ([img, warped],
                  [sharpen1, sharpen2])

    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break