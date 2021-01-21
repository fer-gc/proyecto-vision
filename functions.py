from constants import SPACE_COLORS, COLORS_RGB_SEGMENT
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

def color_segmentation(image, space_color, show=False):
    image_hsv = cv.cvtColor( image, space_color )
    if show: cv.imshow("HSL", image_hsv)
    mask = np.zeros( image_hsv.shape[0:2], dtype=np.uint8 )
    for color in COLORS_RGB_SEGMENT.keys():
        hsv = cv.cvtColor( np.uint8([[ COLORS_RGB_SEGMENT[color] ]]), cv.COLOR_RGB2HSV )[0][0]
        lower = np.uint8( [hsv[0] if hsv[0]-10 > 0 else 0 - 10, 0, 0] )
        upper = np.uint8( [hsv[0] + 10, 255, 255] )
        mask_ = np.array( cv.inRange( image_hsv, lower, upper) )
        mask = cv.bitwise_or( mask, mask_ )
        if show: cv.imshow( color, mask_ )
    mask =  cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS,(15,15)))
    if show: cv.imshow("Final mask", mask)
    return mask

def histogram_equ(imag):
    hist,bins=np.histogram(imag.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[imag]
    plt.subplot(233),plt.imshow(img2),plt.title('Manzana Equalizada')