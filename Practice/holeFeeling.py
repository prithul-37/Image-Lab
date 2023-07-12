import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("img2.jpg",cv.IMREAD_GRAYSCALE)

def onClick(event):
    global x,y 
    ax = event.inaxes
    if ax is not None:
        x,y = ax.transData.inverted().transform([event.x,event.y])
        x = int(round(x))
        y = int(round(y))
        
        print(x,y)
        
        #holefeeling
        emptyImg= np.zeros(img.shape,np.uint8)
        emptyImg[y,x] = 255
        cv.imshow("gg",emptyImg) 
        cv.waitKey(0)
        cv.destroyAllWindows()
        emptyImg_ = np.zeros(img.shape,np.uint8) 
        # cv.imshow("gg",gg) 
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        strElement = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
        
        while(1):
            emptyImg_ = emptyImg
            emptyImg = cv.dilate(emptyImg_,strElement,iterations = 1)
            temp = cv.bitwise_not(img)
            emptyImg = cv.bitwise_and(temp,emptyImg)
            diff = np.array_equal(emptyImg_,emptyImg)
            if diff == True :
                break
        output =   cv.bitwise_or(img,emptyImg)
        cv.imshow("output",output)
        cv.waitKey(0)
        cv.destroyAllWindows() 


plt.title("hello world")
im = plt.imshow(img,cmap='gray')
im.figure.canvas.mpl_connect('button_press_event',onClick)
plt.show()