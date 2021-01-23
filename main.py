#Librerias
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorflow import keras

#Propias
from constants import COLORS_RGB_SEGMENT
from functions import show_space_colors, color_segmentation, histogram_equ, find_contours

#Pruebas
from os import listdir, rename

def main(img):
    image = cv.imread( img )
    if image is None:
        print("Imagen no encontrada, saliendo")
        sys.exit()

    model = keras.models.load_model("apple_classifier")
    print( model.summary() )

    cv.imshow("original", image)

    histogram_equ(image)

    filtered_img = cv.medianBlur(image, 15)
    cv.imshow( "filtro", filtered_img )

    segmented_mask = color_segmentation( filtered_img, cv.COLOR_BGR2HSV )
    segmented_image = cv.bitwise_and( image, image, mask=segmented_mask )
    cv.imshow( "Imagen segmentada", segmented_image )

    rois = find_contours(segmented_mask, image, 0.01)
    cv.imshow("contornos", image)

    #prd = cv.resize( image, (180, 180)  )
    #prd[:,:,3] = np.ones( (180, 180) )

    print( "Prediccion:", model.predict( image ) )
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    test = True
    if test:
        main("./P/image_5.png")
    else:
        directory = "N"
        files = [files for files in listdir( directory ) ]
        for image in files:
            img = f"./{directory}/{image}"
            main(img)