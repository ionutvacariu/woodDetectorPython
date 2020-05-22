import urllib.request
import cv2
import numpy as np
import time

URL = "http://192.168.0.108:8080/video"
cap = cv2.VideoCapture("http://192.168.0.108:8080/video")

while True:
    ret, frame = cap.read()  #
    cv2.imshow("Image", frame)

