import cv2

img = cv2.imread('Process Images/image.jpeg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (4,4), 2)

# Dilating the image a bit.
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))
# dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
edged = cv2.Canny(gray, 0, 84)

cv2.imwrite('Process Images/canny.jpeg', edged)