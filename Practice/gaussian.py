import math
import cv2 as cv
import numpy as np

img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)

def gFilter(sigma,dim):
    kernel = np.zeros((dim,dim))
    print(kernel.shape)
    r= dim//2
    for x in range(-r,r+1):
        for y in range(-r,r+1):
            c= x*x+y*y
            c = c/(sigma*sigma)
            c= math.exp(-c)
            c= c/(2*3.1216*sigma*sigma)
            kernel[x+r,y+r]=c
    return kernel/np.sum(kernel)

gf = gFilter(2,11)
print(gf)
    

def normalization(img):
    out_max = img.max()
    out_min = img.min()
    print(out_max,out_min)
    
    img = img - out_min
    img = img/(out_max - out_min)
    img = img*255
    
    return np.array( img, np.uint8)

def conv(img,kernel):
    h = img.shape[0]
    w = img.shape[1]
    
    out = np.zeros([h,w])
    print(out.shape)
    
    padding_x = kernel.shape[1]//2
    padding_y = kernel.shape[0]//2
    
    borderImage = cv.copyMakeBorder(img,padding_y,padding_y,padding_x,padding_x,cv.BORDER_CONSTANT)
    print(borderImage.shape)
    
    for i in range (padding_y,borderImage.shape[0]-padding_y):
        for j in range (padding_x,borderImage.shape[1]-padding_x):
            temp = 0
            for x in range (-padding_y,padding_y+1):
                for y in range(-padding_x,padding_y+1):
                    temp = temp + kernel[x+padding_y][y+padding_x]* borderImage[i-x][j-y]
            out[i-padding_x][j-padding_y] = temp
    
    ##normalization
    out = normalization(out)
            
    return out

out = conv(img,gf)

cv.imshow("In",img)
cv.imshow("out",out)
cv.waitKey(0)
cv.destroyAllWindows()