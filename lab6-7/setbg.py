# Initialisation
import cv2
import numpy as np

# Compute absolute colour difference of two images.
# The two images must have the same size.
# Return combined absolute difference of the 3 channels 

def absDiff(image1, image2):
	if image1.shape != image2.shape:
		print ('image size mismatch')
		return 0
	else:
		height,width,dummy = image1.shape
		# Compute absolute difference.
		diff = cv2.absdiff(image1, image2)
		a = cv2.split(diff)
		# Sum up the differences of the 3 channels with equal weights.
		# You can change the weights to different values.
		sum = np.zeros((height,width), dtype=np.uint8)
		for i in (1, 2, 3):
			ch = a[i-1]
			cv2.addWeighted(ch, 1.0/i, sum, float(i-1)/i, gamma=0.0, dst=sum)
		return sum
		

def set_background(image, diff, threshold, bgcolor):
	ret, mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
	# mask_inv = cv2.bitwise_not(mask)

	fg_masked = cv2.bitwise_and(image, image, mask=mask)
	fg_masked[np.logical_not(mask)] = bgcolor

	return fg_masked

#-----------------------------
# -----------------------------
# Main

# Initialisation
filename1 = "eagle-1.jpg"
filename2 = "eagle-2.jpg"
difffilename = "eagle-diff.jpg"

# Load images
image1 = cv2.imread(filename1)
image2 = cv2.imread(filename2)

# Compute colour difference and remove background
diff = absDiff(image1, image2)
cv2.imwrite(difffilename, diff)

image_fg = set_background(image2, diff, 10, (0,0,0))
# I see the foreground without the background: the eagle with some noise.
cv2.imwrite('eagle-fg.jpg', image_fg)

traffic1 = cv2.imread("./traffic-1.bmp")
traffic2 = cv2.imread("./traffic-2.bmp")
traffic_diff = absDiff(traffic1, traffic2)
cv2.imwrite("traffic-diff.jpg", traffic_diff)

traffic_fg = set_background(traffic2, traffic_diff, 10, (255,0,0))

# I see the foreground without the background: cars with some noise.
cv2.imwrite("traffic-fg.jpg", traffic_fg)
