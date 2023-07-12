import math
import numpy as np
import cv2 as cv

img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)
outGama = np.zeros(img.shape)
c=.1

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r = (1/c)*(img[i,j]/255)
        r = pow(2.71828,r)-1
        outGama[i][j] = r



def normalization(img):
    out_max = img.max()
    out_min = img.min()
    print(out_max,out_min)
    
    img = img - out_min
    img = img/(out_max - out_min)
    img = img*255
    
    return np.array( img, np.uint8)
      


cv.imshow("out",normalization(outGama))
cv.waitKey(0)
cv.destroyAllWindows()
