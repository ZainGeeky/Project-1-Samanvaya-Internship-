#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('Task_One/Section 1 : Basics/Images/Castle.jpg')
cv.imshow('Castle', img)
cv.waitKey(0)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)
cv.waitKey(0)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)
cv.waitKey(0)

rotated_rotated = rotate(img, -90)
cv.imshow('Rotated Rotated', rotated_rotated)
cv.waitKey(0)

# Resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)
cv.waitKey(0)

# Flipping
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)
cv.waitKey(0)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)