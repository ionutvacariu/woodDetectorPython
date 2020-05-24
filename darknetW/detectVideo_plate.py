import sys, os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import shutil

sys.path.append(os.path.join(os.getcwd(), 'python/'))
import cv2
import numpy as np
import time

TEMP_DETECTED_WOOD_DIRECTORY = 'detectedPlates/temp_img'
DETECTED_WOOD_DIRECTORY = 'detectedPlates/img'

plateClasses = ["plate"]

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
net = cv2.dnn.readNet("../weights/plate_detection_final.weights", "../weights/plate_recognition.cfg")

colors = np.random.uniform(0, 255, size=(len(plateClasses), 3))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

width = int(350)
height = int(100)
video_writer = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc(*'XVID')

dimOfResizedPlatesImage = (width, height)


def moveRemaningFiles(videoName):
    files = os.listdir(TEMP_DETECTED_WOOD_DIRECTORY)
    for f in files:
        if f.__contains__(videoName):
            shutil.move(TEMP_DETECTED_WOOD_DIRECTORY + "/" + f, DETECTED_WOOD_DIRECTORY)


def detect():
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                # onject detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # rectangle co-ordinaters
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x, y, w, h])  # put all rectangle areas
                confidences.append(
                    float(confidence))  # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected
    return confidences, boxes


def addDetectionsToImage(videoName, maxConfidence, confidences, boxes, frame, indexes):
    currentConfidence = -1
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(plateClasses[class_ids[i]])
            confidence = confidences[i]
            currentConfidence = round(confidence, 2)
            color = colors[class_ids[i]]
            crop_img = frame[y:y + h, x:x + w]
            resized = cv2.resize(crop_img, dimOfResizedPlatesImage, interpolation=cv2.INTER_AREA)
            maxConfidence = saveImageWithMaxConfidence(currentConfidence, frame, maxConfidence, resized, videoName)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(currentConfidence), (x, y + 30), font, 1, (255, 255, 255), 2)
            # frameR = cv2.resize(crop_img, dimOfResizedPlatesImage, interpolation=cv2.INTER_AREA)
            video_writer.write(resized)
            print("detected plate on video " + videoName + " Confidence : " + str(
                confidence) + "max confidence till now" + str(maxConfidence))
    return maxConfidence, currentConfidence


def saveImageWithMaxConfidence(currentConfidence, frame_back, maxConfidence, resized, videoName):
    if currentConfidence > maxConfidence:
        cv2.imwrite("detectedPlates/temp_img/detectedPlate" + videoName + "conf" + ".jpg",
                    resized)
        cv2.imwrite("detectedPlates/temp_img/detectedPlate" + videoName + "conf" + "_large.jpg",
                    frame_back)
        maxConfidence = currentConfidence
    return maxConfidence


def substring_after(s, delim):
    return s.partition(delim)[2]


def substring_before(s, delim):
    return s.partition(delim)[0]


def substractVideoName(video_name):
    first = substring_after(video_name, "detectedWood/")
    second = substring_before(first, ".avi")
    return second


def startPlateDetection(wood_detected_video):
    frame_id = 0
    global height, width, outs, class_ids, boxes, maxConfidence
    print("recieved args: " + wood_detected_video)
    detectedWoodVideo = 'detectedPlates/detectedPlates' + str(time.time()) + '.avi'
    video_writer.open(detectedWoodVideo, fourcc, 60.0,
                      (int(width), int(height)),
                      True)
    cap = cv2.VideoCapture(wood_detected_video)
    opened = False;
    maxConfidence = 0;
    videoName = substractVideoName(wood_detected_video)
    while cap.isOpened():
        ret, frame = cap.read()  #
        if ret:
            # cv2.imshow("Image", frame)
            opened = True
            # print("no ca facem afisare")
            frame_id += 1
            height, width, channels = frame.shape
            # detecting objects
            # https://github.com/arunponnusamy/object-detection-opencv/issues/5  --  0.00392  -> 1/255
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # reduce 416 to 320

            net.setInput(blob)
            outs = net.forward(outputlayers)

            # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
            class_ids = []
            boxes = []
            (confidences, boxes) = detect()
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
            maxConfidence, confidence = addDetectionsToImage(videoName, maxConfidence, confidences, boxes,
                                                             frame,
                                                             indexes)

            key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame
            if key == 27:  # esc key stops the process
                break;
        else:
            if opened:
                print("finished to detect on " + wood_detected_video)
            else:
                print("can't open " + wood_detected_video)
            break;
    moveRemaningFiles(videoName)
    cap.release()
    video_writer.release()
    print("max confidence: " + str(maxConfidence))
    if not maxConfidence > 0:  # if max confidence > 0, we detected at least 1 frame
        os.remove(detectedWoodVideo)
        print("removed " + detectedWoodVideo)
    cv2.destroyAllWindows()


# startPlateDetection("detectedWood/plates2.mp4")
class ExampleHandler(FileSystemEventHandler):
    def on_created(self, event):  # when file is created
        # do something, eg. call your function to process the image
        pathToWoodVideo = event.src_path
        # time.sleep(15)
        print("Got event for file %s" % pathToWoodVideo)
        t = threading.Thread(target=startPlateDetection(pathToWoodVideo),
                             name="startingPlateDetection")
        t.daemon = True
        t.start()
        # startPlateDetection(pathToWoodVideo)


observer = Observer()
event_handler = ExampleHandler()  # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path='detectedWood')
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
