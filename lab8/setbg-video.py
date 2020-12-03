import cv2
import numpy as np


shiTo_params = dict( maxCorners = 40,
                       qualityLevel = 0.03,
                       minDistance = 10,
                       blockSize = 7 )

lk_params = dict( winSize  = (30,30),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def process(video):
    # number of frames :
    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Initialize everything with the first frame.
    ret, previous_frame = video.read()
    gray_previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Write the background removed video
    # Warning: don't forget that opencv is height/width (shape), so we have to inverse things here...
    # The joy of opencv I guess :)
    out = cv2.VideoWriter('result-2.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps, (previous_frame.shape[1], previous_frame.shape[0]))

    # define some parameters.
    optical_flow_params = dict(winSize=(30,30),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               100, 0.3))

    feature_to_track_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=50,
                          blockSize=8)

    old_points = cv2.goodFeaturesToTrack(gray_previous_frame, mask=None, **feature_to_track_params)

    # well, we worked on the first frame (0) so we're on the second frame now.
    current_frame = 1

    # Create a mask image for drawing purposes
    mask = np.zeros_like(previous_frame)

    while current_frame < total_frames:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_frame = current_frame +1

        if frame is None:
            continue

        # Gaussian Blur to remove high frequency noise.
        blur = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        # Apply a threshold to improve the car edge detection.
        ret, image_thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # Dilate a bit to improve the car edge detection.
        kernel_dilatation = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(image_thresh, kernel_dilatation, iterations=2)

        # Perform Canny edge detection
        edged_image = cv2.Canny(dilated, 5, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        closed = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        p_new, status, error = cv2.calcOpticalFlowPyrLK(gray_previous_frame, gray_frame, old_points, None, **optical_flow_params)

        if p_new is not None:
            good_new = p_new[status == 1]
            good_old = old_points[status == 1]
        else:
            print('<!!> not able to run optical flow.')
            continue


        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # try linear regression...
            # model = LinearRegression().fit(a, b)
            # r_sq = model.score(a, b)
            # Predict next X coord, with given Y
            # I can't make it... huh :( :(

            from scipy.spatial import distance
            aa = (a, b)
            bb = (c, d)
            dst = distance.euclidean(aa, bb)

            if dst < 60:
                # Add also a function that drawn a line between each of the previous feature locations to
                # depict this movement.
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
            # else:
            #     print(f'dst={dst}')

            frame = cv2.circle(frame, (int(a), int(b)), 5, (0,0,255), -1)

        img = cv2.add(frame, mask)

        out.write(img)

        gray_previous_frame = gray_frame.copy()
        old_points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_to_track_params)

    video.release()
    out.release()

if __name__ == '__main__':
    print('run main')

    video = cv2.VideoCapture('Results_Background.avi')
    process(video)
