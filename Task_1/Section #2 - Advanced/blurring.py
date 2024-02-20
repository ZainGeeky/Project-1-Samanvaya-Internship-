#pylint:disable=no-member

import cv2 as cv

img = cv.imread('Task_One/Resources/Photos/cats.jpg')
cv.imshow('Cats', img)
cv.waitKey(0)

# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)
cv.waitKey(0)

# Gaussian Blur
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)
cv.waitKey(0)

# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)
cv.waitKey(0)

# Bilateral
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)