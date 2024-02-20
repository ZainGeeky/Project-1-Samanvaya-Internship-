#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('Task_One/Resources/Photos/park.jpg')
cv.imshow('Park', img)
cv.waitKey(0)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])


cv.imshow('Blue', blue)
cv.waitKey(0)
cv.imshow('Green', green)
cv.waitKey(0)
cv.imshow('Red', red)
cv.waitKey(0)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)