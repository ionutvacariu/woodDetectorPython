import sys, os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.join(os.getcwd(), 'python/'))
import cv2
import numpy as np
import time
import threading

from characterRecognition.kafka_producer_plate_number import sendMess

classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    woodClasses = f.read().rstrip('\n').split('\n')

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

net = cv2.dnn.readNet("char_final.weights", "char.cfg")

#net = cv2.dnn.readNet("char_yolov3_final_yolov4.weights", "char.cfg")
colors = np.random.uniform(0, 255, size=(len(woodClasses), 3))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def findInDetection(outs1, width, height):
    global confidence, w, h, x, y, indexes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs1:
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
                class_ids.append([class_id, x])  # name of the object tha was detected
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    return confidences, boxes, class_ids


def printDetectionAndSave(confidences1, boxes, class_ids, img1):
    global x, y, w, h, confidence
    ordered_boxes, ordered_cls = orderFromLeftToRight(boxes, class_ids)
    licensePlate = "";
    for i in range(len(ordered_boxes)):
        if i in indexes:
            x, y, w, h = ordered_boxes[i]
            cls_id, x = ordered_cls[i]
            label = str(woodClasses[cls_id])
            confidence = confidences1[i]
            color = colors[cls_id]
            # print(label)
            licensePlate += label
            cv2.rectangle(img1, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img1, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)
            cv2.imwrite("result_character_plate.jpg", img1)

    return licensePlate;


def orderFromLeftToRight(boxes, class_ids):
    ordered_boxes = sorted(boxes, key=lambda tup: tup[0])
    ordered_cls = sorted(class_ids, key=lambda tup: tup[1])
    return ordered_boxes, ordered_cls


def start(image):
    img = cv2.imread(image)

    width = int(400)
    height = int(100)
    dimOfResizedPlatesImage = (width, height)

    img = cv2.resize(img, dimOfResizedPlatesImage, interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # reduce 416 to 320
    net.setInput(blob)
    outs = net.forward(outputlayers)
    confidences, boxes, class_ids = findInDetection(outs, width, height)
    licensePlate = printDetectionAndSave(confidences, boxes, class_ids, img)
    if len(licensePlate) == 0:
        licensePlate = "unidentified"
    print("numar gasit " + licensePlate)
    #sendMess(licensePlate, image)


# start("inmatriculare/4.jpg")

from characterRecognition.mimetypesUtilities import IMAGE


class ExampleHandler(FileSystemEventHandler):
    def on_created(self, event):  # when file is created
        # do something, eg. call your function to process the image
        pathToWoodVideo = event.src_path
        # time.sleep(15)
        filename, file_extension = os.path.splitext(pathToWoodVideo)
        if IMAGE.__contains__(file_extension) and not filename.__contains__("_large"):
            print("Got event for file %s" % pathToWoodVideo)
            t = threading.Thread(target=start(pathToWoodVideo),
                                 name="startingPlateDetection")
            t.daemon = True
            t.start()


observer = Observer()
event_handler = ExampleHandler()  # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path='../darknetW/detectedPlates/img')
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
