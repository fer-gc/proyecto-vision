import cv2
import matplotlib.pyplot as plt
import numpy as np


img=cv2.imread('C:\\Users\\Usuario\\Documents\\Primer semestre Post Polonia\\VA\\Proyecto Final reconocimiento de fruta\\Images\\N\\Manchada.jpg')
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Conversión de BGR a RGB
img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY) # Conversión escala de grises

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def Histo(imag):
    hist,bins=np.histogram(imag.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[imag]
    plt.subplot(233),plt.imshow(img2),plt.title('Manzana Equalizada')
    

Histo(img_rgb)
plt.subplot(231),plt.imshow(img_rgb),plt.title('Original')
plt.subplot(232),plt.imshow(img_gray,cmap='gray'),plt.title('Grayscale')
plt.subplot(234),plt.imshow(img_hsv,cmap='gray'),plt.title('HSV Scale')
plt.show()


"""
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

blurred_imag=cv2.blur(gray,(3,3))
cv2.imshow('filtrada',blurred_imag)
cv2.waitKey(0)
"""