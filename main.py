
import os
import json
#import tensorflow as tf
import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
import pytesseract
import text_detection
import testing

_test_image = "C:\\Users\\ajatkins\\Desktop\\OCR models\\test_image\\Screenshot016.png"

if __name__ == '__main__':

    #test
    timer = testing.Timer()

    #Load config file [TODO: error handling]
    with open(os.path.join(os.path.dirname(__file__),"config.json")) as config_fp:
        config = json.load(config_fp)

    pytesseract.pytesseract.tesseract_cmd = config["tesseract_cmd"]


    #OpenCV
    text_detector = text_detection.CV_Text_Detector(
                                                    config["cv_text_detector_model"],
                                                    config["cv_image_resolution_W"],
                                                    config["cv_image_resolution_H"]
                                                   )

    [original,detection_boxes] = text_detector.detect_image(_test_image)
    text = []
    with timer:
        for box in detection_boxes:
            #cropped
            text.append(pytesseract.image_to_string(original[int(box[0][1]):int(box[1][1]),
                                                             int(box[0][0]):int(box[2][0])]))
            #decoded
    print(timer.duration())

    print(text)
    testing.display(detection_boxes,original)
