# coding: utf-8
import cv2
import numpy as np

## ---------------------------------------------------------------------
## 3.1  Extract SURF keypoints and descriptors from an image. ----------
def extract_features_and_descriptors(image):

    ## Convert image to grayscale (for ORB detector).
    ## DONE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## Detect ORB features and compute descriptors.
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(gray_image, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(gray_image, kp)

    return kp, des

## --------------------------------------------------------------------
def detect_features(grey_image):
    '''Example: perform detection of key-points only'''
    orb = cv2.ORB_create()
    return orb.detect(grey_image)

def extract_descriptors(grey_image, keypoints):
    '''Example: calculate descriptor using existing keypoints'''
    orb = cv2.ORB_create()
    return orb.compute(grey_image, keypoints)[1]


## --------------------------------------------------------------------
## 3.2 Find corresponding features between the images. ----------------
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):

    ## Find corresponding features.
    matches = match_flann(descriptors1, descriptors2)

    print(f'matches={matches}')

    ## Look up corresponding keypoints.
    points1 = []
    points2 = []
    ## DONE

    for match in matches:
        p1 = keypoints1[match.queryIdx].pt
        p2 = keypoints2[match.trainIdx].pt

        points1.append(p1)
        points2.append(p2)

    return points1, points2


## ---------------------------------------------------------------------
## 3.3  Calculate the size and offset of the stitched panorama. --------
def calculate_size(size_image1, size_image2, homography):
    ## Calculate the size and offset of the stitched panorama.

    print(f'size_image2 shape={size_image2[0]}, {size_image2[1]} ')

    """
    homography is a 3*3 matrix. So I create a vector of 3 with [x,y,1] where the "1" won't impact anything.
    """
    x = size_image2[0]
    y = size_image2[1]

    vector = (x, y, 1)

    """
    extract result: 0:2 -> to take the first two rows which corresponds to x,y.
    On these rows, we take the last value (2) to extract x offset and y offset.
    
    [ 1 , 0 , x_offset]
    [ 0 , 1 , y_offset]
    [ 0 , 0 ,    1    ]
    """
    offset = abs((homography * vector)[0:2, 2])
    size = (size_image1[1] + int(offset[0]) + 30, size_image1[0] + int(offset[1]) + 50)

    # reset offset
    offset[0] = 0
    offset[1] = 0

    print(f'calculate_size: offset={offset} size={size}')

    ## Calculate the size and offset of the stitched panorama.
    #
    # ## Update the homography to shift by the offset
    homography[0,2] += offset[0]
    homography[1,2] += offset[1]

    return size, offset


## ---------------------------------------------------------------------
## 3.4  Combine images into a panorama. --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):
    ## Combine the two images into one.
    panorama = np.zeros((size[1], size[0], 3), np.uint8)

    image2 = cv2.warpPerspective(image2, homography, size)

    print(f'offset={offset} size={size} image1 shape: {image1.shape} image2 shape: {image2.shape}')

    place_image(panorama, image2, offset[0], offset[1])
    place_image(panorama, image1, offset[0], offset[1])

    # cv2.drawChessboardCorners(panorama, size, keypoints, False)

    return panorama


def place_image(output, image, x, y):
    minx = max(x,0)
    miny = max(y,0)
    maxx = min(x+image.shape[1],output.shape[1])
    maxy = min(y+image.shape[0],output.shape[0])
    output[miny:maxy, minx:maxx] = image[miny-y:maxy-y, minx-x:maxx-x]


def match_flann(desc1, desc2, r_threshold = 0.7):
    # Finds strong corresponding features in the two given vectors.

    if len(desc1) == 0 or len(desc2) == 0:
        print ("No features passed into match_flann")
        return []

    flann = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    # TODO: follow
    #  https://docs.opencv.org/4.4.0/d5/d6f/tutorial_feature_flann_matcher.html
    # use the knnMatch to find matching pairs
    matched_pairs = flann.knnMatch(desc1, desc2, 2)

    # TODO: filter closest pairs using the distance threshold
    #  https://docs.opencv.org/4.4.0/d5/d6f/tutorial_feature_flann_matcher.html
    strong_pairs = []

    for m, n in matched_pairs:
        if m.distance < r_threshold * n.distance:
            strong_pairs.append(m)

    ## Only return robust feature pairs.
    return strong_pairs


def draw_correspondences(image1, image2, points1, points2):
    'Connects corresponding features in the two images using yellow lines.'

    ## Put images side-by-side into 'image'.
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image[:h1, :w1] = image1
    image[:h2, w1:w1+w2] = image2

    ## Draw yellow lines connecting corresponding features.
    for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
        cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.LINE_AA)

    return image

## ---------------------------------------------------------------------

def show(name, im):
    if im.dtype == np.complex128:
        raise Exception("OpenCV can't operate on complex valued images")

    cv2.namedWindow(name)
    cv2.imshow(name, im)
    cv2.waitKey(1)

if __name__ == "__main__":
    print('run main')

    ## Load images.
    image1 = cv2.imread("input/Image1.jpg")
    image2 = cv2.imread("input/Image2.jpg")

    print(f'shape of image1: {image1.shape}')

    ## Detect features and compute descriptors.
    keypoints1, descriptors1 = extract_features_and_descriptors(image1)
    keypoints2, descriptors2 = extract_features_and_descriptors(image2)
    print(len(keypoints1), "features detected in image1")
    print(len(keypoints2), "features detected in image2")

    img1_out = np.zeros_like(image1)
    img2_out = np.zeros_like(image2)
    show("Image1 features", cv2.drawKeypoints(image1, keypoints1, img1_out, color=(0,0,255)))
    show("Image2 features", cv2.drawKeypoints(image2, keypoints2, img2_out, color=(0,0,255)))

    ## Find corresponding features.
    points1, points2 = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
    points1 = np.array(points1, dtype=float)
    points2 = np.array(points2, dtype=float)
    print (len(points1), "features matched")

    ## Visualise corresponding features.
    correspondences = draw_correspondences(image1, image2, points1, points2)
    cv2.imwrite("correspondences.jpg", correspondences)
    cv2.imshow('correspondences', correspondences)

    ## Find homography between the views.
    if len(points1) < 4 or len(points2) < 4:
        print ("Not enough features to find a homography")
        homography = np.identity(3, dtype=float)
    else:
        (homography, _) = cv2.findHomography(points2, points1, method=cv2.RANSAC)
        # homography = np.matrix(homography)
    print ("Homography = ")
    print (homography)

    ## Calculate size and offset of merged panorama.
    (size, offset) = calculate_size(image1.shape[:2], image2.shape[:2], homography)
    size = tuple(np.asarray(size).flatten().astype(int).tolist())
    offset = tuple(np.asarray(offset).flatten().astype(int).tolist())
    print ("output size: %ix%i" % size)

    ## Finally combine images into a panorama.
    panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
    cv2.imwrite("panorama.jpg", panorama)
    cv2.imshow('panorama', panorama)


    import sys, select
    print ("Press enter or any key on one of the images to exit")
    while True:
        if cv2.waitKey(100) != -1:
            break
        # http://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
        i, o, e = select.select([sys.stdin], [], [], 0.1 )
        if i:
            break
