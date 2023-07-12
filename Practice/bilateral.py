import math
import cv2 as cv
import numpy as np

def normalization(img):
    img_min = img.min()
    img_max = img.max()
    
    img = img - img_min
    img = img/(img_max-img_min)
    
    for i in range (0,img.shape[0]):
        for j in range (0,img.shape[1]):
            img[i,j]=round(img[i,j]*255)
            
    return np.array(img,np.uint8)

img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)
ker1 = np.zeros((5,5))

for i in range(0,ker1.shape[0]):
    for j in range(0,ker1.shape[1]):
        dis = (i-2)*(i-2) + (j-2)*(j-2)
        ker1[i,j] = dis
#print(ker1)
ker1 = ker1/np.sum(ker1)

def conv(img,ker):
    padx=ker.shape[1]//2
    pady=ker.shape[0]//2
    
    borderImage = cv.copyMakeBorder(img,padx,padx,pady,pady,cv.BORDER_CONSTANT)
    out = np.zeros(img.shape)
    
    for x in range(padx,borderImage.shape[0]-padx):
        for y in range(padx,borderImage.shape[0]-padx):
            tempKer = np.zeros(ker.shape)
            for i in range (-padx,padx+1):
                for j in range (-pady,pady+1):
                    gg = borderImage[x,y]- borderImage[x+i,y+j]
                    gg = -(gg*gg)/(2*5*5)
                    gg = math.exp(gg)
                    tempKer[i+padx,j+pady] = x
            tempKer = tempKer*ker
            tempKer = tempKer/np.sum(tempKer)
            temp = 0 
            for i in range (-padx,padx+1):
                for j in range (-pady,pady+1):
                    temp += tempKer[i+padx,j+pady]*borderImage[x-i,y-j]
            out[x-padx,y-pady] = temp
            
    out  = normalization(out)
    cv.imshow("gg",out)
    return out
                    
cv.imshow("input",img)         
cv.imshow("outPut",conv(img,ker1))
cv.waitKey(0)
cv.destroyAllWindows()