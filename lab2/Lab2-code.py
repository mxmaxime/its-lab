"""
Please read "result.md
"""
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def inverse(imagem):
    return 255 - imagem


def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1, image2).any())


if __name__ == '__main__':
    """
    Question 1
    """
    scale_img = cv2.imread(r'scale.jpg', cv2.IMREAD_GRAYSCALE)
    print('matrix values of scale.jpg')
    print(scale_img)
    print(scale_img.shape)


    """
    Question 2
    """
    image = cv2.imread(r'Lena.jpg')

    # Transform image color to gray scales
    gs_imagem = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # call the function inverse
    inverse_img = inverse(gs_imagem)
    cv2.imwrite("Lena1.jpg", inverse_img)


    """
    Question 3
    """
    lenaA = cv2.imread(r'Lena-A.jpg')
    lenaB = cv2.imread(r'Lena-B.jpg')

    is_equals = np.all(lenaA == lenaB)
    print(f'is lenaA identical to lenaB? {is_similar(lenaA, lenaB)}')
    print(f'is lenaA identical to lenaB? {is_equals}')

    inverse_lena_a = inverse(lenaA)
    inverse_lena_b = inverse(lenaB)

    sum_inverse = (inverse_lena_a /2) + (inverse_lena_b/2)
    cv2.imwrite("sum_inverse.jpg", sum_inverse)

    is_equals = np.all(lenaA == lenaB)
    print(is_similar(lenaA, lenaB))

