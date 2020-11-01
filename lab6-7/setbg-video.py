import cv2
import numpy as np


# Compute absolute colour difference of two images.
# The two images must have the same size.
# Return combined absolute difference of the 3 channels
def abs_diff(image1, image2):
    if image1.shape != image2.shape:
        print('image size mismatch')
        return 0
    else:
        height, width, dummy = image1.shape
        # Compute absolute difference.
        diff = cv2.absdiff(image1, image2)
        a = cv2.split(diff)
        # Sum up the differences of the 3 channels with equal weights.
        # You can change the weights to different values.
        sum = np.zeros((height, width), dtype=np.uint8)

        for i in (1, 2, 3):
            ch = a[i - 1]
            cv2.addWeighted(ch, 1.0 / i, sum, float(i - 1) / i, gamma=0.0, dst=sum)

        return sum


def set_background(image, diff, threshold, bgcolor):
    _, mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)

    # get foreground thanks to the mask.
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # change the background: everything that is not the foreground mask is the background.
    foreground[np.logical_not(mask)] = bgcolor

    return foreground

def average(video, sec = 0):
    if sec == 0:
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(fps * sec)

    print(f'total_frames={total_frames}')
    ret, first_frame = video.read()
    avg = first_frame

    gamma = 0

    alpha = 0.90
    gap = 1 - alpha
    incr = gap / total_frames

    current_frame_number = 0

    while current_frame_number <= total_frames:
        # print(f'{current_frame_number} = current_frame_number')

        ret, frame = video.read()
        current_frame_number = current_frame_number+1

        if frame is None:
            continue

        alpha += incr

        # should not happen but it does sometimes... With the lost of precision.
        if alpha > 1:
            alpha = 1

        # by definition, alpha + beta = 1
        beta = 1 - alpha

        avg = cv2.addWeighted(avg, alpha, frame, beta, gamma)

    video.release()
    return avg

def remove_background(video, avg, sec = 0):
    # number of frames :
    fps = video.get(cv2.CAP_PROP_FPS)

    if sec == 0:
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        total_frames = int(fps * sec)

    # Write the background removed video
    # Warning: don't forget that opencv is height/width (shape), so we have to inverse things here...
    # The joy of opencv I guess :)
    out = cv2.VideoWriter('result.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps, (avg.shape[1], avg.shape[0]))

    current_frame = 0
    while current_frame < total_frames:
        ret, frame = video.read()
        current_frame = current_frame +1
        # print(f'{current_frame} = current_frame')

        if frame is None:
            continue

        # 1) Call absDiff to compute the difference between the video frame and the
        diff = abs_diff(avg, frame)

        # 2) Call setBackground to set the background of the video frame to green (0, 255, 0)
        # with an appropriate threshold.
        without_background = set_background(frame, diff, 68, (0, 255, 0))

        # 3) Save the background-removed video frame into the output video.
        out.write(without_background)

        if current_frame == 1:
            print('save the first background-removed video frame into an output image file.')
            # 4) Save the first background-removed video frame into an output image file.
            cv2.imwrite("first_background_removed.jpg", without_background)

    video.release()
    out.release()

if __name__ == '__main__':
    print('run main')

    nb_sec = 3
    video = cv2.VideoCapture('traffic.avi')
    avg = average(video, nb_sec)
    cv2.imwrite('avg_background.jpg', avg)

    video = cv2.VideoCapture('traffic.avi')
    remove_background(video, avg, nb_sec)
