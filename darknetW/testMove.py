import os
import shutil

TEMP_DETECTED_WOOD_DIRECTORY = 'detectedPlates/temp_img'
DETECTED_WOOD_DIRECTORY = 'detectedPlates/img'


def moveRemaningFiles(videoName):
    files = os.listdir(TEMP_DETECTED_WOOD_DIRECTORY)
    for f in files:
        if f.__contains__(videoName):
            shutil.move(TEMP_DETECTED_WOOD_DIRECTORY + "/" + f, DETECTED_WOOD_DIRECTORY)


moveRemaningFiles("plates2.mp4")
