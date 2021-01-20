
COLORS_RGB_SEGMENT = {
    'rojo': [122, 2, 8],
    'verde': [246, 255, 163],
    'blanco': [209, 153, 158]
}
SPACE_COLORS = {
    "HSV": "COLOR_BGR2HSV",
    "HLS": "COLOR_BGR2HLS",
    "RGB": "COLOR_BGR2RGB"
}

if __name__ == "__main__":
    import cv2 as cv
    import numpy as np

    print("Constantes necesarias para el proyecto")
    print( "*" * 50 )
    print("Colores a segmentar:")
    for color in COLORS_RGB_SEGMENT.keys():
        low, high = COLORS_RGB_SEGMENT[color]
        rgb_low = np.uint8([[ low ]])
        rgb_high = np.uint8([[ high ]])
        hsv_low = cv.cvtColor( rgb_low , cv.COLOR_BGR2HSV )
        hsv_high = cv.cvtColor( rgb_high , cv.COLOR_BGR2HSV )
        print(f"{color}_low:\t{rgb_low} RGB\t{hsv_low} HSV")
        print(f"{color}_high:\t{rgb_high} RGB\t{hsv_high} HSV")