#!/usr/bin/env python
 
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
import time
 
cap = cv2.VideoCapture('video.mp4')
 
ret,previous = cap.read()
best_frame = previous
number = 50
file = 0
 
while(True):
    i = 0
    best_s = 0
    while(i<=number):
    i += 1
    ret, frame = cap.read()
    s = ssim(frame, previous, multichannel=True)
    if(s > best_s):
        best_s = s
        best_frame = frame
    cv2.imwrite('folder/%d.png' % file,best_frame)
    file += 1
    previous = best_frame
 
cap.release()
cv2.destroyAllWindows()