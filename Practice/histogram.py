import copy
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("Lena.jpg")
imgG  = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)
outNOR = np.zeros(img.shape, np.uint8)

print(img.shape)
cv.imshow("out",img)
cv.waitKey(0)

outImg = np.zeros(img.shape,np.uint8)
outImg2 = np.zeros(img.shape,np.uint8)
outImg3 = np.zeros(img.shape,np.uint8)

# hist1 = cv.calcHist(img)
# hist2 = cv.calcHist(img[:,:,1])
# hist3 = cv.calcHist(img[:,:,2])


img1 = copy.deepcopy(img[:,:,0])
img2 = copy.deepcopy(img[:,:,1])
img3 = copy.deepcopy(img[:,:,2])

def calHist(imgg):
    hist = [0 for i in range (0,256)]
    for x in range (0,imgg.shape[0]):
        for y in range (0,imgg.shape[0]):
            hist[imgg[x,y]] +=1
    return hist

def eqialization(arr,imgg):
    
    
    out = copy.deepcopy(imgg)
    totalPixel = imgg.shape[0]*imgg.shape[1]
    
    hist = arr
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



hits1 = calHist(img1)
hits2 = calHist(img2)
hits3 = calHist(img3)


outImg[:,:,0] = img1
outImg2[:,:,1] = img2
outImg3[:,:,2] = img3

cv.imshow("out1",outImg)
cv.imshow("out2",outImg2)
cv.imshow("out3",outImg3)
cv.waitKey(0)

hist4 = calHist(imgG)


norImg1 = eqialization(hits1,img1)
norImg2 = eqialization(hits2,img2)
norImg3 = eqialization(hits3,img3)

outImg[:,:,0] = norImg1
outImg2[:,:,1] = norImg2
outImg3[:,:,2] = norImg3


outNOR[:,:,0] = norImg1
outNOR[:,:,1] = norImg2
outNOR[:,:,2] = norImg3

cv.imshow("out",img)
cv.imshow("out1",outImg)
cv.imshow("out2",outImg2)
cv.imshow("out3",outImg3)
cv.imshow("outG",imgG)
cv.imshow("outNORM",outNOR)


# plt.figure(figsize=(10,10))

# plt.subplot(2, 2, 1)
# plt.title("Line graph blue")
# plt.hist(img[:,:,0].ravel(),256,[0,255],color = "blue")

# plt.subplot(2, 2, 2)
# plt.title("Line graph green")
# plt.hist(img[:,:,1].ravel(),256,[0,255],color = "green")

# plt.subplot(2, 2, 3)
# plt.title("Line graph red")
# plt.hist(img[:,:,2].ravel(),256,[0,255],color = "red")

# plt.subplot(2, 2, 4)
# plt.title("Line graph green")
# plt.hist(imgG.ravel(),256,[0,255],color = "gray")

# plt.show()


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.title("Line graph1")
plt.plot(hits1, color="blue")

plt.subplot(2,2,2)
plt.title("Line graph2")
plt.plot(hits2, color="green")

plt.subplot(2,2,3)
plt.title("Line graph3")
plt.plot(hits3, color="red")

plt.subplot(2,2,4)
plt.title("Line graph4")
plt.plot(hist4, color="gray")
plt.show()

cv.waitKey(0)
cv.destroyWindow()