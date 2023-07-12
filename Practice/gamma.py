import numpy as np
import cv2 as cv

img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)
outGama = np.zeros(img.shape)



for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r = img[i,j]/255
        y = .1
        r = pow(r,y)
        outGama[i][j] = r*255
        
out = np.array(outGama, np.uint8)

cv.imshow("out",out)
cv.waitKey(0)
cv.destroyAllWindows()
