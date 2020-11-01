"""
Please read "result.md
"""

import cv2
import numpy as np


if __name__ == '__main__':
    image = cv2.imread(r'YywIS.jpg')

    lower_red = np.array([150, 0, 50])
    upper_red = np.array([180, 255, 255])

    lower_red2 = np.array([0, 0, 50])
    upper_red2 = np.array([30, 255, 255])

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])


    # do the color conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # merge the two red masks.
    red_mask = red_mask | red_mask2

    final_mask = blue_mask + red_mask + green_mask

    #save the images
    cv2.imwrite('HSV.jpg', hsv)
    cv2.imwrite('blue_mask.jpg', blue_mask)
    cv2.imwrite('red_mask.jpg', red_mask)
    cv2.imwrite('green_mask.jpg', green_mask)
    cv2.imwrite('final_mask.jpg', final_mask)
