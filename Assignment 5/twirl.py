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

cv2.imshow("Imsg",img)
cv2.waitKey(0)

(x,y)=(250,250)
red = int(input("R:"))
a=int(input("Angle:"))
a = math.radians(a)

out = np.zeros((500,500),np.uint8)


for i in range (0,500):
    for j in range(0,500):
        dx = x-i
        dy = y-j 
        red2 = red*red
        dis = dx*dx+dy*dy
        
        if(dis>red2):
            out[i][j] = img[i][j]
        else:
            dis =  math. sqrt(dis)
            bta = math.atan2(dy, dx)+a*((red-dis)/red)
            
            nx = x + dis * math.cos(bta)
            ny = y + dis * math.sin(bta)
            #print(nx,ny)
            out[i][j] = img[int(nx)][round(ny)]
        
cv2.imshow("Imsg",out)
cv2.waitKey(0)
cv2.destroyAllWindows();