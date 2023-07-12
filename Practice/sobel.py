import cv2
import numpy 


img = cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)


# kernel1 = numpy.ones((25,25))
# kernel1 = kernel1/(25*25)

kernel1 = numpy.array([[-1,0,1],
                       [-1,0,1],
                       [-1,0,1]])

kernel2 = numpy.array([[-1,-1,-1],
                       [0,0,0],
                       [1,1,1]])

def normalization(img):
    out_max = img.max()
    out_min = img.min()
    print(out_max,out_min)
    
    img = img - out_min
    img = img/(out_max - out_min)
    img = img*255
    
    return numpy.array( img, numpy.uint8)

def conv(img,kernel):
    h = img.shape[0]
    w = img.shape[1]
    
    out = numpy.zeros([h,w])
    print(out.shape)
    
    padding_x = kernel.shape[1]//2
    padding_y = kernel.shape[0]//2
    
    borderImage = cv2.copyMakeBorder(img,padding_y,padding_y,padding_x,padding_x,cv2.BORDER_CONSTANT)
    print(borderImage.shape)
    
    for i in range (padding_y,borderImage.shape[0]-padding_y):
        for j in range (padding_x,borderImage.shape[1]-padding_x):
            temp = 0
            for x in range (-padding_y,padding_y+1):
                for y in range(-padding_x,padding_y+1):
                    temp = temp + kernel[x+padding_y][y+padding_x]* borderImage[i-x][j-y]
            out[i-padding_x][j-padding_y] = temp
    
    ##normalization
    
            
    return out





out1 = conv(img,kernel1)
out2 = conv(img,kernel2)

out3 = out1*out1 + out2*out2
out3 = numpy.sqrt(out3)

cv2.imshow("in",img)
cv2.imshow("out1",normalization(out1))
cv2.imshow("out2",normalization(out2))
cv2.imshow("out3",normalization(out3))

cv2.waitKey(0)
cv2.destroyAllWindows()