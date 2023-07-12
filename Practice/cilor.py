import copy
import cv2 as cv
import numpy as np


img = cv.imread("Lena.jpg")
out = np.zeros(img.shape,np.uint8)

img1 = img[:,:,0]
img2 = img[:,:,1]
img3 = img[:,:,2]


def eqialization(imgg):
    
    
    out = copy.deepcopy(imgg)
    totalPixel = imgg.shape[0]*imgg.shape[1]
    
    hist = [0 for i in range(0,256)]
    
    for x in range (0,imgg.shape[0]):
        for y in range (0,imgg.shape[0]):
            hist[imgg[x,y]]+=1
            
    pdf = [0 for i in range(0,256)]
    cdf = [0 for i in range(0,256)]
    c = [0 for i in range(0,256)]
    newHist = [0 for i in range(0,256)] 
    
    for i in range(0,256):
        pdf[i] = hist[i]/totalPixel
    
    cdf[0] = pdf[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]
        c[i] = round(cdf[i]*255)
    
    for x in range (0,imgg.shape[0]):
        for y in range (0,imgg.shape[0]):
            out[x][y] = c[imgg[x,y]]
    return out


out[:,:,0] = eqialization(img1)
out[:,:,1] = eqialization(img2)
out[:,:,2] = eqialization(img3)

cv.imshow("out",out)
cv.waitKey(0)
cv.destroyAllWindows()



