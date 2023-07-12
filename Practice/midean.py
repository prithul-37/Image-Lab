import cv2 as cv
import numpy as np


img = cv.imread("median2.jpg",cv.IMREAD_GRAYSCALE)

def median(x,y,img):
    padx = y//2
    pady = x//2
    
    boederedImage = cv.copyMakeBorder(img,padx,padx,pady,pady,cv.BORDER_CONSTANT)
    
    out = np.zeros(img.shape)
    
    mid = (x*y)//2
    
    for i in range(padx,boederedImage.shape[0]-padx):
        for j in range(pady,boederedImage.shape[1]-pady):
            temp = []
            for x in range(-padx,padx+1):
                for y in range(-pady,pady+1):
                    temp.append(boederedImage[x+i,y+j])
            temp.sort()
            out[i-padx,j-pady]=temp[mid]
            
    return np.array(out,np.uint8)

out = median(5,5,img)

cv.imshow("in",img)
cv.imshow("out",out)
cv.waitKey(0)
cv.destroyWindow()