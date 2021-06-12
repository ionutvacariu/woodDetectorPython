import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python/'))
import cv2
import numpy as np
import time
import shutil
from collections import namedtuple

CUTOUT_WOOD_WITH_TIME = '/cutout_wood_withTime'
TEMP_DETECTED_WOOD_DIRECTORY = 'tempDetectedWood'
DETECTED_WOOD_DIRECTORY = 'detectedWood'

woodClasses = ["wood"]
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
net = cv2.dnn.readNet("../weights/wood.weights", "../weights/wood.cfg")
colors = np.random.uniform(0, 255, size=(len(woodClasses), 3))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture("wood4.mp4")
cap = cv2.VideoCapture("input_video/20210605_121628.mp4")
#cap = cv2.VideoCapture("20200606_133924.mp4")
# cap = cv2.VideoCapture(0)  # 0 for 1st webcam

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
dimOfResizedPlatesImage = (width, height)
Detection = namedtuple("Detection", ["frame", "prediction"])
max_width = int(0)
max_height = int(0)
frames = []

print("width_of_video " + str(width) + "height_of_vide" + str(height))
out_video = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case

isStarted = False


def findInDetection():
    global confidence, w, h, x, y, indexes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
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
    global x, y, w, h, confidence, isStarted, timer_start, out_video_withtime, detectedWoodVideo, max_width, max_height, frames
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
            print(str(confidence))

    # cv2.imshow("ImageWOOD", frame)
    # frameR = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def saveFramesAsVideo(frames, max_height, max_width):
    global out_video_withtime, detectedWoodVideo, isStarted
    out_video_withtime = cv2.VideoWriter()

    detectedWoodVideo = CUTOUT_WOOD_WITH_TIME + str(time.time()) + '.mov'
    print("created video " + detectedWoodVideo)
    out_video_withtime.open(TEMP_DETECTED_WOOD_DIRECTORY + detectedWoodVideo, fourcc, 30.0,
                            (int(max_width), int(max_height)),
                            True)
    if not( not frames):
        for cropped_frame in frames:
            resized = resizeWithPadding(cropped_frame)
            out_video_withtime.write(resized)  # save frame in new video
            cv2.imshow("ImageWOOD", resized)

        print("start detecting plate on video " + detectedWoodVideo)
        isStarted = False
        out_video_withtime.release()
        os.replace(TEMP_DETECTED_WOOD_DIRECTORY + detectedWoodVideo,
                   DETECTED_WOOD_DIRECTORY + '/' + detectedWoodVideo)


def resizeWithPadding(img):
    h, w = img.shape[:2]
    diff_vert = max_height - h
    pad_top = diff_vert // 2
    pad_bottom = diff_vert - pad_top
    diff_hori = max_width - w
    pad_left = diff_hori // 2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert img_padded.shape[:2] == (max_height, max_width)
    return img_padded

def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),-min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def rotateImage(image, angle):
    if hasattr(image, 'shape'):
        image_center = tuple(np.array(image.shape) / 2)
        shape = image.shape
    elif hasattr(image, 'width') and hasattr(image, 'height'):
        image_center = (image.width / 2, image.height / 2)
        shape = np.array((image.width, image.height))
    else:
        raise Exception('Unable to acquire dimensions of image for type %s.' % (type(image),))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, shape, flags=cv2.INTER_LINEAR)
    return result


def startDetector():
    global frame, outs, class_ids, confidences, boxes, frame_id
    while cap.isOpened():
        ret, frame = cap.read()  #
        if ret:
            frame_id += 1
            # if height < width:
            #     frame = imutils.rotate(frame, 0)
            # height, width, channels = frame.shape

            # blobFromImage (InputArray image, double scalefactor=1.0, const Size &size=Size(), const Scalar &mean=Scalar(), bool swapRB=false, bool crop=false, int ddepth=CV_32F)
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True,
                                         crop=False)

            net.setInput(blob)
            # detecting objects
            outs = net.forward(outputlayers)
            # print(outs[1])
            class_ids = []
            confidences = []
            boxes = []

            findInDetection()

            printDetectionAndSave()

            key = cv2.waitKey(33)  # wait 1ms the loop will start again and we will process the next frame

        else:
            break;
    #saveFramesAsVideo(frames, max_height, max_width)

    moveRemaningFiles()

    print("DONE")
    cap.release()
    cv2.destroyAllWindows()


def moveRemaningFiles():
    files = os.listdir(TEMP_DETECTED_WOOD_DIRECTORY)
    for f in files:
        shutil.move(TEMP_DETECTED_WOOD_DIRECTORY + "/" + f, DETECTED_WOOD_DIRECTORY)


startDetector()
