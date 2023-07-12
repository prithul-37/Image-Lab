import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def normalization(img):
    out_max = img.max()
    out_min = img.min()
    print(out_max,out_min)
    
    img = img - out_min
    img = img/(out_max - out_min)
    img = img*255
    
    return np.array( img, np.uint8)

img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)

ft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(ft)

mag1 = np.abs(fft_shift)
mag2 = 1*np.log(mag1+1)
mag3 = normalization(mag2)

angle = np.angle(fft_shift)

final = mag1*np.exp(1j*angle)
final = np.real(np.fft.ifft2(np.fft.ifftshift(final)))
final = normalization(final)

cv.imshow("Outmag",mag3)
cv.imshow("Outang",normalization(angle))
cv.imshow("Out",final)
cv.waitKey(0)
cv.destroyAllWindows()