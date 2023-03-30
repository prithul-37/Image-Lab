
import numpy as np
import cv2
import math
img = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
out=img.copy()

#cv2.imshow('input image',img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i,j)
        d=a/32
        c=pow(2,d)
        c=c-1
        out.itemset((i,j),c)
        
cv2.imshow('output image',out)

cv2.imshow('input image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()