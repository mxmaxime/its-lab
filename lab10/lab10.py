import cv2
import time
import numpy as np


def detect_faces(img, xml):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(xml)

    scale_factor = 1.250
    minNeighbors = 1
    faces = faceCascade.detectMultiScale(gray, scale_factor, minNeighbors)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img

def detect_webcam_faces(xml):
    webcam = cv2.VideoCapture(0)
    _, video_frame = webcam.read()

    while webcam.isOpened():
        _, video_frame = webcam.read()
        frame_faces = detect_faces(video_frame,xml)
        cv2.imshow("Webcam face detection using opencv", frame_faces)

        # esc touch
        key = cv2.waitKey(2)
        if key == 27:
            break

    # don't forget to release the webcam, you could lock it and have to restart the computer... :)
    webcam.release()


img = cv2.imread("ubuntu_team.jpg")
xml = "haarcascade_frontalface.xml"

def first_part():
    faces = detect_faces(img, xml)
    cv2.imwrite("faces.jpg", faces)
    cv2.imshow("Faces", faces)
    cv2.waitKey()

def second_part():
    detect_webcam_faces(xml)

first_part()
# second_part()
