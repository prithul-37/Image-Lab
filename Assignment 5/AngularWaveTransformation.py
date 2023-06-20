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
                    
(x,y)=(0,0)
tau = 100
a= 10
out = np.zeros((525,525),np.uint8)


for i in range (0,500):
    for j in range(0,500):
        dx = x-i
        dy = y-j 
        dis = dx*dx+dy*dy
        r = math.sqrt(dis)
        
        nx = i + 10*math.sin((2*3.1416*j)/120)
        ny = j + 15*math.sin((2*3.1416*i)/250)
        
        out[(int(nx)+12)%525][(int(ny)+12)%525] = img[i][j]
        
        


cv2.imshow("Imsg",out)
cv2.waitKey(0)
cv2.destroyAllWindows();