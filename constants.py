
COLORS_RGB_SEGMENT = {
    'rojo': [138, 4, 5],
    'brillo': [191, 96, 102],
    "moradp": [ 33, 0, 17 ]
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
        clr = COLORS_RGB_SEGMENT[color]
        rgb = np.uint8([[ clr ]])
        hsv = cv.cvtColor( rgb, cv.COLOR_BGR2HSV )
        print(f"{color}_low:\t{rgb} RGB\t{hsv} HSV")