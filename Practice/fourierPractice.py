import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

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

ft = np.fft.fft2(img)
fftss = np.fft.fftshift(ft)

mag1 = np.abs(fftss)
mag2 = 1*np.log(mag1+1)
mag3 = normalization(mag2)

angle = np.angle(fftss)

final = mag1*np.exp(1j*angle)
final = np.real(np.fft.ifft2(np.fft.ifftshift(final)))
final = normalization(final)


cv.imshow("In",img)
cv.imshow("mag",mag3)
cv.imshow("angle",angle)
cv.imshow("out",final)
cv.waitKey(0)
cv.destroyAllWindows()
