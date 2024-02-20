import cv2
import os

# Ensure correct path to Haar cascade file
haar_cascade_path = os.path.join('Task_1', 'Section #3 - Faces', 'haar_face.xml')
haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

img = cv2.imread('Task_1/Resources/Photos/group 2.jpg')
cv2.imshow('Group of 5 people', img)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray People', gray)
cv2.waitKey(0)

# Adjust scale factor and minimum neighbors if needed
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f'Number of faces found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
