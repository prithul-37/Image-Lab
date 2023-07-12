import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from copy import deepcopy as dpc

D0 = 5 #radius
n = 2 #order
def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x,y))

def FilterMake(filter, uk, vk):
    M = filter.shape[0]
    N = filter.shape[1]
    H = np.ones((M,N), np.float32)
    for u in range(M):
        for v in range(N):
            H[u, v] = 1.0
            dk = ((u - M//2-uk)**2 + (v -N//2-vk)**2)**(0.5)
            d_k = ((u - M//2+uk)**2 + (v -N//2+vk)**2)**(0.5)
            if dk==0 or d_k==0:
                H[u, v] = 0.0
            else:
                H[u, v] = (1/(1+((D0/dk)**(2*n)))) * (1/(1+((D0/d_k)**(2*n))))
    return H

def butter(filter, points):
    ret = np.ones(filter.shape, np.float32)
    for u,v in points:  
        ret *= FilterMake(filter, v, u)
    return ret


# Start From Here
img_input = cv2.imread('te.jpg', cv2.IMREAD_GRAYSCALE)
img = dpc(img_input)
image_size = img.shape[0] * img.shape[1]
M=img.shape[0]
N=img.shape[1]

filterr=np.zeros((M,N),np.float32)

# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum_ac=np.abs(ft_shift)
magnitude_spectrum = 1 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)
ang = np.angle(ft_shift)

point_list=[]
img = np.zeros((10,12), np.uint8)
img[4:6, 5:7] = 1
x = None
y = None
X = np.zeros_like(img)

plt.title("Please select seed pixel from the input")
im = plt.imshow(magnitude_spectrum, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

print(point_list)
M = img_input.shape[0]
N = img_input.shape[1]
filter = np.zeros((M,N), np.float32())
filter = butter(filter, point_list)
print(filter)

filtered_img= ft *filter
cv2.imshow("Spectrum Image", min_max_normalize(filtered_img))
## phase add
filtered_img = np.multiply(filtered_img, np.exp(1j*ang))
# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_img)))
img_back= min_max_normalize(img_back)

cv2.imshow("Input Image", img_input)
##cv2.imshow("Spectrum Image", magnitude_spectrum_scaled)
cv2.imshow("Output Image", img_back)


cv2.waitKey(0)
cv2.destroyAllWindows() 



