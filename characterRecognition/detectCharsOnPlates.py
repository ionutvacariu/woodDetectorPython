import sys, os

sys.path.append(os.path.join(os.getcwd(), 'python/'))
import cv2
import numpy as np
import time

classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    woodClasses = f.read().rstrip('\n').split('\n')

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
net = cv2.dnn.readNet("char.weights", "char.cfg")

colors = np.random.uniform(0, 255, size=(len(woodClasses), 3))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("imageBre1589040703.66871.jpg")
blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

net.setInput(blob)
outs = net.forward(outputlayers)

width = int(350)
height = int(100)

class_ids = []
confidences = []
boxes = []

licensePlate = ""


def findInDetection():
    global confidence, w, h, x, y, indexes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
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
    global x, y, w, h, confidence, licensePlate
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(woodClasses[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            # print(label)
            licensePlate += label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)
            cv2.imwrite("result_character_plate.jpg", img)
    cv2.imshow("Image", img)


def start():
    findInDetection()
    printDetectionAndSave()
    print("numar gasit " + licensePlate)


start()
