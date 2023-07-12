import cv2
import numpy 


img = cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)


# kernel1 = numpy.ones((25,25))
# kernel1 = kernel1/(25*25)

kernel1 = numpy.array([[0,-1,0],
                       [-1,4,-1],
                       [0,-1,0]])


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
    out = normalization(out)
            
    return out


def corr(img,kernel):
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
                    temp = temp + kernel[x+padding_y][y+padding_x]* borderImage[i+x][j+y]
            out[i-padding_x][j-padding_y] = temp
    
    ##normalization
    # out_max = out.max()
    # out_min = out.min()
    # print(out_max,out_min)
    # for i in range(0,out.shape[0]):
    #     for j in range(0,out.shape[1]):
    #         temp = out[i][j]
    #         temp -= out_min
    #         temp /= out_max - out_min
    #         out[i][j] = round(temp*255)
     
    normalization(out)
            
    return out


out = conv(img,kernel1)
cv2.imshow("in",img)
cv2.imshow("out",out)

cv2.waitKey(0)
cv2.destroyAllWindows()