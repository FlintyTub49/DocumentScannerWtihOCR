import cv2

img = cv2.imread('../Doc Scanner/Process Images/IMG_0227.jpg', 1) 

lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)
cv2.imwrite('../Doc Scanner/Process Images/contrast2.jpg', final)

cv2.waitKey(0)