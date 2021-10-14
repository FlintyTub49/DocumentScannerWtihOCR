import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cap = cv2.VideoCapture(1)
cap.set(10, 150)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1200, 675))
    img = cv2.imread('../Doc Scanner/Process Images/contrast2.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # gray = cv2.GaussianBlur(gray, (7,7), 5)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # edged = cv2.Canny(dilated, 0, 90)
    # cv2.imshow('Dilated', thresh_img)
    # cv2.imshow('Canny', edged)

    (cnts, hierarchy) = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for i in range(len(cnts)):
        cv2.drawContours(img, cnts, i, (0, 230, 255), 6)

    cv2.imshow('Extreme Cases', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
