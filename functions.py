from constants import SPACE_COLORS
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def show_space_colors(img_bgr):
    for j, space_name in enumerate(SPACE_COLORS.keys()):
        color = list(space_name)
        image = cv.cvtColor( img_bgr, eval(f"cv.{SPACE_COLORS[space_name]}") )
        plt.subplot(int( f"31{j+1}" ))
        for i, col in enumerate(color):
            histr = cv.calcHist([image],[i],None,[256],[0,256])
            plt.plot(histr)
            plt.legend(color)
            plt.title(space_name)
    plt.show()