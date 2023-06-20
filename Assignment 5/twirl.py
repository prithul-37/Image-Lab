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
red = 245
a=90
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
            if(dx==0):
                bta = 90+a*((red-dis)/red)
            else:
                bta = math.degrees(math.atan(dy/dx))+a*((red-dis)/red)
            
            if(dx>=0 and dy>=0):
                bta = bta
            if(dx<=0 and dy>=0):
                bta =  bta
            if(dx<=0 and dy<=0):
                bta =  bta + 180
            if(dx>=0 and dy<=0):
                bta =  bta + 180
            
            nx = x + dis * math.cos(math.radians(bta))
            ny = y + dis * math.sin(math.radians(bta))
            #print(nx,ny)
            out[round(nx)][round(ny)] = img[i][j]
        
        


cv2.imshow("Imsg",out)
cv2.waitKey(0)
cv2.destroyAllWindows();