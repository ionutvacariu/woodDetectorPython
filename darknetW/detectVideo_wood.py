import sys, os

CUTOUT_WOOD_WITH_TIME = '/cutout_wood_withTime'
TEMP_DETECTED_WOOD_DIRECTORY = 'tempDetectedWood'
DETECTED_WOOD_DIRECTORY = 'detectedWood'

sys.path.append(os.path.join(os.getcwd(), 'python/'))
import cv2
import numpy as np
import time
import threading
import shutil

# import darknetW.detectVideo_plate as plateRecognition

woodClasses = ["wood detector"]

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
net = cv2.dnn.readNet("../weights/wood.weights", "../weights/wood.cfg")

colors = np.random.uniform(0, 255, size=(len(woodClasses), 3))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture("wood4.mp4")
cap = cv2.VideoCapture("wood8.mov")
# cap = cv2.VideoCapture(0)  # 0 for 1st webcam

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("width_of_video " + str(width) + "height_of_vide" + str(height))

out_video = cv2.VideoWriter()
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

# capSize = (1028,720) # this is the size of my source video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case

# success = self.vout.open('output.mov',fourcc,30,capSize,True)


# out_video.open('cutout_wood ' + str(time.time()) + '.avi', fourcc, 60.0, (int(width), int(height)), True)

isStarted = False


def findInDetection():
    global confidence, w, h, x, y, indexes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # onject detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                # rectangle co-ordinaters
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x, y, w, h])  # put all rectangle areas
                confidences.append(
                    float(confidence))  # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)


def printDetectionAndSave():
    global x, y, w, h, confidence, isStarted, timer_start, out_video_withtime, detectedWoodVideo
    for i in range(len(boxes)):
        if i in indexes:

            if not isStarted:
                timer_start = time.time()
                isStarted = True
                out_video_withtime = cv2.VideoWriter()

                detectedWoodVideo = CUTOUT_WOOD_WITH_TIME + str(time.time()) + '.mov'
                print("created video " + detectedWoodVideo)
                out_video_withtime.open(TEMP_DETECTED_WOOD_DIRECTORY + detectedWoodVideo, fourcc, 30.0,
                                        (int(width), int(height)),
                                        True)

            start = time.time() - timer_start
            if start > 150:
                print("start detecting plate on video " + detectedWoodVideo)
                isStarted = False
                out_video_withtime.release()
                os.replace(TEMP_DETECTED_WOOD_DIRECTORY + detectedWoodVideo,
                           DETECTED_WOOD_DIRECTORY + '/' + detectedWoodVideo)

            x, y, w, h = boxes[i]
            label = str(woodClasses[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            out_video_withtime.write(frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)
    # frameR = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow("ImageWOOD", frame)


def startDetector():
    global frame, outs, class_ids, confidences, boxes, frame_id
    while cap.isOpened():
        ret, frame = cap.read()  #
        if ret:
            frame_id += 1

            # height, width, channels = frame.shape
            # detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True,
                                         crop=False)  # reduce 416 to 320

            net.setInput(blob)
            outs = net.forward(outputlayers)
            # print(outs[1])

            # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
            class_ids = []
            confidences = []
            boxes = []

            findInDetection()

            printDetectionAndSave()

            elapsed_time = time.time() - starting_time

            key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

            if key == 27:  # esc key stops the process
                break;
        else:
            break;

    moveRemaningFiles()

    print("DONE")
    cap.release()
    cv2.destroyAllWindows()


def moveRemaningFiles():
    files = os.listdir(TEMP_DETECTED_WOOD_DIRECTORY)
    for f in files:
        shutil.move(TEMP_DETECTED_WOOD_DIRECTORY + "/" + f, DETECTED_WOOD_DIRECTORY)


startDetector()

# out_video.release()
