import cv2
import numpy as np


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('motion.avi')

# Params to get good features to track
nb_corners_to_track = 40
quality_level = 0.03
min_distance_between_corners = 10

frame_counter = 0

old_levels_points = {
    'level_0': [],
    'level_1': [],
    'level_2': [],
}

new_level_points = {
    'level_0': [],
    'level_1': [],
    'level_2': [],
}

old_frame = None

while True:
    _, frame = cap.read()
    if frame is None:
        print('reset video')
        # time.sleep(5)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if old_frame is None:
        old_frame = new_frame.copy()
        continue

    # 1)
    _, pyr_new_frame = cv2.buildOpticalFlowPyramid(new_frame, winSize=(30, 30), maxLevel=3)

    # Extracts first and sec level
    new_pyr_level_2 = pyr_new_frame[6]
    new_pyr_level_1 = pyr_new_frame[4]
    new_pyr_level_0 = pyr_new_frame[2]

    new_pyr_levels = [new_frame, new_pyr_level_1, new_pyr_level_2]

    _, pyr_old_frame = cv2.buildOpticalFlowPyramid(old_frame, winSize=(30, 30), maxLevel=3)

    last_pyr_level_2 = pyr_old_frame[6]
    last_pyr_level_1 = pyr_old_frame[4]
    last_pyr_level_0 = pyr_old_frame[2]

    last_pyr_levels = [old_frame, last_pyr_level_1, last_pyr_level_2]

    for i, (last_pyr_level, new_pyr_level) in enumerate(zip(last_pyr_levels, new_pyr_levels)):
        new_points = cv2.goodFeaturesToTrack(last_pyr_level, nb_corners_to_track, quality_level, min_distance_between_corners, False)

        old_points = old_levels_points[f'level_{i}']

        # First frame initialization.
        if len(old_points) == 0:
            old_levels_points[f'level_{i}'] = new_points
            print('first frame init points for {i} level')
        else:
            # 2)
            points, status, error = cv2.calcOpticalFlowPyrLK(last_pyr_level, new_pyr_level, old_points, None)
            new_level_points[f'level_{i}'] = points
            old_levels_points[f'level_{i}'] = points


    old_frame = new_frame.copy()

    all_points = np.concatenate([new_level_points[key] * 2**i for i, key in enumerate(new_level_points.keys())])

    for new in all_points:
        a,b = new.ravel()


        frame = cv2.circle(frame, (a,b), 5, (0,0,255), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()