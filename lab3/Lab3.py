import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    image = cv2.imread('nature.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
    cv2.destroyAllWindows()
