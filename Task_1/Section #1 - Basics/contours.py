#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('Task_One/Section 1 : Basics/Images/cats.jpg')
cv.imshow('Cats', img)
cv.waitKey(0)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)
cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
cv.waitKey(0)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)
cv.waitKey(0)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)
cv.waitKey(0)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)