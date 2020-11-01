# motion tracking by Lucas-Kanade tracker.
import cv2

# Read two images.
im1 = cv2.imread("motion0025.jpg")
im2 = cv2.imread("motion0026.jpg")
gs_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gs_im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
width, height = gs_im1.shape

# Get good features to track
# These parameters seem's good.
nb_corners_to_track = 40
quality_level = 0.03
min_distance_between_corners = 10
p0 = cv2.goodFeaturesToTrack(gs_im1, nb_corners_to_track, quality_level, min_distance_between_corners, False)

# Call tracker.
p1, st, err = cv2.calcOpticalFlowPyrLK(gs_im1, gs_im2, p0, None, (3,3))
print(len(p1))

for i,(new,old) in enumerate(zip(p1,p0)):
    a,b = new.ravel()
    c,d = old.ravel()
    im1 = cv2.circle(im1, (c,d), 3, (255,255,255), -1)
    im2 = cv2.circle(im2, (a,b), 3, (255,255,255), -1)

print(len(p1))

cv2.imwrite("track0025.jpg", im1)
cv2.imwrite("track0026.jpg", im2)
