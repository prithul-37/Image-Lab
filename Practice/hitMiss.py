import cv2
import numpy as np

img = cv2.imread('gg.jpeg', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)
cv2.imshow("Original", img)

kernel1=np.array(([0,0,0],
          [1,1,0],
          [1,0,0]), np.uint8)

kernel2=np.array(([0,1,1],
          [0,0,1],
          [0,0,1]), np.uint8)

kernel3=np.array(([1,1,1],
          [0,1,0],
          [0,1,0]), np.uint8)

W=np.array(([1,1,1],
          [1,1,1],
          [1,1,1]), dtype='uint8')

rate = 50
W = cv2.resize(W, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
kernel1=cv2.resize(kernel1, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
kernel2=cv2.resize(kernel2, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
kernel3= cv2.resize(kernel3, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)


W = W *255
kernel1 = kernel1 *255
kernel2 = kernel2 *255
kernel3 = kernel3 *255

# cv2.imshow("W",W)
# cv2.imshow("kernel1",kernel1)
# cv2.imshow("kernel2",kernel2)
# cv2.imshow("kernel3",kernel3)
# cv2.waitKey(0)


B2=cv2.subtract(W,kernel1)
B3=cv2.subtract(W,kernel2)
B4=cv2.subtract(W,kernel3)

cv2.imshow("B2",B2)
cv2.imshow("b3",B3)
cv2.imshow("b4",B4)
cv2.waitKey(0)




output1=np.ones(img.shape,np.uint8)
output2=np.ones(img.shape,np.uint8)
output3=np.ones(img.shape,np.uint8)

com=np.ones(img.shape,np.uint8)
com=cv2.bitwise_not(img)

a = cv2.erode(img,kernel1,iterations = 1)
cv2.imshow("a",a)

b= cv2.erode(com,B2, 1)
cv2.imshow("b",b)

output1=cv2.bitwise_and(a,b)
#output1=cv2.dilate(output1,,iterations=1)

cv2.imshow("output1",output1)

c= cv2.erode(img,kernel2,iterations = 1)

d= cv2.erode(com,B3,iterations = 1)

output2=cv2.bitwise_and(c,d)
cv2.imshow("output2",output2)


e = cv2.erode(img,kernel3,iterations = 1)

f= cv2.erode(com,B4,iterations = 1)
output3=cv2.bitwise_and(e,f)

cv2.imshow("output3",output3)


cv2.waitKey(0)
cv2.destroyAllWindows()