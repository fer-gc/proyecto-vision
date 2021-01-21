#Librerias
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
#import tensorflow as tf

#Propias
from constants import COLORS_RGB_SEGMENT
from functions import show_space_colors, color_segmentation, histogram_equ

#Pruebas
from os import listdir, rename



def main(img):
    image = cv.imread( img )
    if image is None:
        sys.exit()

    cv.imshow("original", image)

    histogram_equ(image)

    filtered_img = cv.medianBlur(image, 15)
    cv.imshow( "filtro", filtered_img )

    segmented_mask = color_segmentation( filtered_img, cv.COLOR_BGR2HSV )
    segmented_image = cv.bitwise_and( image, image, mask=segmented_mask )
    cv.imshow( "Imagen segmentada", segmented_image )

    contours, hierarchy = cv.findContours( segmented_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
    approx = []
    for cnt in contours:
        approx.append( cv.convexHull(cnt) )

    cv.drawContours(image, approx, -1, (0,255,0), 3)
    cv.imshow("contornos", image)


    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    test = False
    if test:
        main("./P/image_0.jpg")
    else:
        directory = "P"
        files = [files for files in listdir( directory ) ]
        for image in files:
            img = f"./{directory}/{image}"
            main(img)