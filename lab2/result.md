Please, see the code also :)

## Question 1
```python
scale_img = cv2.imread(r'scale.jpg', cv2.IMREAD_GRAYSCALE)
print('matrix values of scale.jpg')
print(scale_img)
```

Note: thanks to the parameter `cv2.IMREAD_GRAYSCALE` we obtain a matrix of two dimensions and not 3 dimensions, because the color depth of graycale can be put in a unique value, unlink BRG or RGB that needs 3 values.

## Question 2
```python
def inverse(imagem):
    return 255 - imagem

image = cv2.imread(r'Lena.jpg')
gs_imagem = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

inverse_img = inverse(gs_imagem)
cv2.imwrite("Lena1.jpg", inverse_img)
```

## Question 3
To know if pictures `lena-A` and `lena-B` are identical, let create a little function that will compare the two pixel by pixel.

```python
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1, image2).any())

# We also could use the `np.all` numpy function to check if all values are identical.
def is_similar_2(image1, image2):
    return image1.shape == image2.shape and np.all(image1 == image2)
```


And by applying this function, we know that pictures **are not identical**.
```python
lenaA = cv2.imread(r'Lena-A.jpg')
lenaB = cv2.imread(r'Lena-B.jpg')

print(f'is lenaA identical to lenaB? {is_similar(lenaA, lenaB)}') # -> False
print(f'is lenaA identical to lenaB? {is_similar_2(lenaA, lenaB)}') # -> False
```

And let me explain why does it works.
So, first of all, as I'm doing a strict comparaison pixel by pixel, I need the two pictures to have the same shape: same height, width and also the same color channel.

Then, I use the `bitwise_xor` operator, let's take a look of how does it works:
```python
tt = [[1, 3, 160]]
tt2 = [[1, 6, 160]]
print('test bitwise xor: ', np.bitwise_xor(tt, tt2))
# -> test bitwise xor:  [[0 5 0]]
```

It actually does a XOR (exclusive OR): compares two binary numbers bitwise, which means that we perform an XOR operator bit per bit of numbers. So here for instance, `1 ^ 1` equals 0 because bits are equals. But `(3) 011 ^ 110 (6)` output `(5) 101`.


So this is interresting in our case because If two pixels are not equals, then we have a number different of 0. That is why we apply `.any()` method on top of it.

## Question 4

```python
inverse_lena_a = inverse(lenaA)
inverse_lena_b = inverse(lenaB)

sum_inverse = (inverse_lena_a /2) + (inverse_lena_b/2)
cv2.imwrite("sum_inverse.jpg", sum_inverse)
```

## Question 5
Yes, the result does look like Lena-inversed in question 2, but maybe not pixel by pixel.

## Question 6
Please see the code, `Image_color.py`.
Created images:
- `red_mask.jpg`: red circles
- `blue_mask.jpg`: blue circles
- `green_mask.jpg`: green circles
- `final_mask.jpg`: the mask with all the extracted circles, red blue and green.

To extract red circles, I use two mask to have a better results. This is because Red is at the begining of the hsv "map" and also at the end. At the end, I merge the two mask thanks to an OR operator.
