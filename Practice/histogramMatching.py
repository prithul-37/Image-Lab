import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 

img1 = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)
img2 = cv.imread("img.jpg",cv.IMREAD_GRAYSCALE)
out = np.zeros(img1.shape,np.uint8)


def pdf(imgg):
    hist = [0 for i in range (0,256)]
    for x in range (0,imgg.shape[0]):
        for y in range (0,imgg.shape[0]):
            hist[imgg[x,y]] +=1
    
    
    totalPixel= imgg.shape[0] * imgg.shape[1]
    
    
    pdf = [0 for i in range(0,256)]
    cdf = [0 for i in range(0,256)]
    c = [0 for i in range(0,256)]
    newHist = [0 for i in range(0,256)] 
    
    for i in range(0,256):
        pdf[i] = hist[i]/totalPixel
    return pdf

def cdf(imgg):
    
    hist = [0 for i in range (0,256)]
    for x in range (0,imgg.shape[0]):
        for y in range (0,imgg.shape[0]):
            hist[imgg[x,y]] +=1
    
    
    totalPixel= imgg.shape[0] * imgg.shape[1]
    
    
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
    
    
    return c

#image 1 cdf 

cdf1= cdf(img1)

#image 2 cdf

cdf2= cdf(img2)

map = [0 for i in range(0,256)]

print(cdf1)
print(cdf2)

for i in range (0,256):
    x = cdf1[i]
    for j in range(0,256):
        if(cdf2[j]>=x):
            map[i] = j
            s = j
            break;
#print(map)
#map

for i in range(0,img1.shape[0]):
    for j in range(0,img1.shape[1]):
        out[i,j] = map[img1[i,j]]
         
        
plt.figure(figsize=(10,10))

plt.subplot(3,3,1)
plt.imshow(img1,cmap='gray')
plt.title("Imahge 1")

plt.subplot(3,3,2)
plt.hist(img1.ravel(),256,[0,256])
plt.title("Histogram Imahge 1")


plt.subplot(3,3,3)
plt.plot(pdf(img1))
plt.title("Imahge 2")

plt.subplot(3,3,4)
plt.imshow(img2,cmap='gray')
plt.title("Imahge 2")

plt.subplot(3,3,5)
plt.hist(img2.ravel(),256,[0,256])
plt.title("Histogram Imahge 2")

plt.subplot(3,3,6)
plt.plot(pdf(img2))
plt.title("Out")

plt.subplot(3,3,7)
plt.imshow(out,cmap='gray')
plt.title("Out")

plt.subplot(3,3,8)
plt.hist(out.ravel(),256,[0,256])
plt.title("Histogram out")

plt.subplot(3,3,9)
plt.plot(pdf(out))
plt.title("Out")

plt.show()