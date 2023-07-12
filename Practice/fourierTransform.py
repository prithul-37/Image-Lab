import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread("period_input.jpg",cv.IMREAD_GRAYSCALE)


def normalization(img):
    out_max = img.max()
    out_min = img.min()
    print(out_max,out_min)
    
    img = img - out_min
    img = img/(out_max - out_min)
    img = img*255
    
    return np.array( img, np.uint8)
def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')


ft = np.fft.fft2(img)

fftss = np.fft.fftshift(ft)

mg1 = np.abs(fftss)
mg2 = 1*np.log(np.abs(fftss)+1)
mg3 = min_max_normalize(mg2)

filter_ = np.ones(img.shape)

def onClick(event):
    global x,y 
    ax = event.inaxes
    if ax is not None:
        x,y = ax.transData.inverted().transform([event.x,event.y])
        x = int(round(x))
        y = int(round(y))
        print(x,y)
        
        
        #r = int(input("R:"))
    r = 10
        
    for i in range (-r,r+1):
        for j in range (-r,r+1):
            dis = (0-i)**2+(0-j)**2
            if(dis<=r**2):
                filter_[y+i][x+j]=0
        
    cv.imshow("filter",filter_)


plt.title("Please select seed pixel from the input")
im = plt.imshow(mg3, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onClick)
plt.show()

angle = np.angle(fftss)



cv.imshow("filter1",filter_)

mod = np.multiply(filter_,mg1)

cv.imshow("mod",normalization(mg3*filter_))

finalResult = mod*np.exp(1j*angle)

img_back = np.real(np.fft.ifft2(np.fft.ifftshift(finalResult)))
img_back_scaled = min_max_normalize(img_back)


cv.imshow("in",img)
cv.imshow("Final",img_back_scaled)
cv.imshow("Out",mg3)
cv.imshow("phase",angle)
cv.waitKey(0)
cv.destroyAllWindow()


