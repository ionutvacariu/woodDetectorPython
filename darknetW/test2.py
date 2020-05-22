import sys, os

sys.path.append(os.path.join(os.getcwd(), 'python/'))
import cv2
import numpy as np
import time

woodClasses = ["plate"]

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
net = cv2.dnn.readNet("plate_detection_final.weights", "plate_recognition.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

colors = np.random.uniform(0, 255, size=(len(woodClasses), 3))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture("plate_test.mov")

cap = cv2.VideoCapture("cutout_wood_withTime1589951741.699192.avi")
# cap = cv2.VideoCapture("http://192.168.0.108:8080/video")
# cap = cv2.VideoCapture(0)  # 0 for 1st webcam

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(350)
height = int(100)
video_writer = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc(*'XVID')

dim = (width, height)


# resize image

def detect():
    global confidence, w, h, x, y
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


def printImage(videoName):
    global x, y, w, h, confidence
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(woodClasses[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            crop_img = frame[y:y + h, x:x + w]
            resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)
            frameR = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
            video_writer.write(resized)
            print("detected image ")
    cv2.imshow("Image", frame)

    # cv2.imwrite("imageBre" + str(time.time()) + ".jpg", resized)


def startPlateDetection(wood_detected_video):
    global frame, height, width, outs, class_ids, confidences, boxes, indexes, frame_id
    print("recieved args: " + wood_detected_video)
    detectedWoodVideo = '/mydrive/plateDetection/detectedPlates' + str(time.time()) + '.avi'
    video_writer.open(detectedWoodVideo, fourcc, 60.0,
                      (int(width), int(height)),
                      True)
    cap = cv2.VideoCapture(wood_detected_video)
    while cap.isOpened():
        ret, frame = cap.read()  #

        if ret:
            frame_id += 1
            height, width, channels = frame.shape
            # detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

            net.setInput(blob)
            outs = net.forward(outputlayers)
            # print(outs[1])

            # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
            class_ids = []
            confidences = []
            boxes = []
            detect()

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

            printImage(wood_detected_video)

            key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame
            if key == 27:  # esc key stops the process
                break;
        else:
            print("can't open " + wood_detected_video)
            break;
    print("finished to detect on " + wood_detected_video)
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


startPlateDetection('detectedWood/cutout_wood_withTime1589952758.741605.avi')
