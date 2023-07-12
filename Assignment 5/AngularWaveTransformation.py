import math
import cv2 
import numpy as np


img = np.zeros((500,500),np.uint8)


for i in range (0,10):
    for j in range(0,10):
        
        if( (i+j)%2 == 0):
            
            for x in range (0,50):
                for y in range(0,50):
                    img[i*50+x][j*50+y]=255 
                    
(x,y)=(250,250)
tau = int(input("TAU:"))
a= float(input("Amplitude:"))
out = np.zeros((500,500),np.uint8)


for i in range (0,500):
    for j in range(0,500):
        dx = x - i
        dy = y - j
        r = math.sqrt(dx*dx+dy*dy)
        bta =   math.atan2(dx, dy) + a*math.sin(2*math.pi*r/tau)
        
        nx = 250 + r*math.cos(bta)
        ny = 250 + r*math.sin(bta)
        
        nx = int(nx)
        ny = int(ny)
        
        
        
        if nx >= 0 and ny >= 0 and nx < img.shape[0] and ny < img.shape[1]:
            out[i][j] = img[nx,ny]
        
        


cv2.imshow("Imsg",out)
cv2.waitKey(0)
cv2.destroyAllWindows();