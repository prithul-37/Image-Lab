import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)

def onClick(event):
    global x,y 
    ax = event.inaxes
    if ax is not None:
        x,y = ax.transData.inverted().transform([event.x,event.y])
        x = int(round(x))
        y = int(round(y))
        
        print(x,y)


plt.title("hello world")
im = plt.imshow(img,cmap='gray')
im.figure.canvas.mpl_connect('button_press_event',onClick)
plt.show()