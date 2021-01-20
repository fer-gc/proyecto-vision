#Librerias
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf

#Propias
from constants import COLORS_RGB_SEGMENT
from functions import show_space_colors

#Pruebas
from os import listdir, rename

def main(img):
    image = cv.imread( img )
    cv.imshow("originial", image)
    #show_space_colors(image)

    image_hsv = cv.cvtColor( image, cv.COLOR_BGR2HSV )
    cv.imshow("HSL", image_hsv)

    mask = np.zeros( image_hsv.shape[0:2], dtype=np.uint8 )
    for color in COLORS_RGB_SEGMENT.keys():
        hsv = cv.cvtColor( np.uint8([[ COLORS_RGB_SEGMENT[color] ]]), cv.COLOR_RGB2HSV )[0][0]
        lower = np.uint8( [hsv[0] if hsv[0]-10 > 0 else 0 - 10, 0, 0] )
        upper = np.uint8( [hsv[0] + 10, 255, 255] )
        mask_ = np.array( cv.inRange( image_hsv, lower, upper) )
        mask = cv.bitwise_or( mask, mask_ )
        cv.imshow( color, mask_ )
    
    cv.imshow("Final mask", mask)
    apple = cv.bitwise_and( image, image, mask=mask )
    cv.imshow("apple", apple)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    directory = "P"
    files = [files for files in listdir( directory ) ]
    for image in files:
        img = f"./{directory}/{image}"
        main(img)