
import os
import json
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
    results = text_detection.tesseract_experience_value([[original,detection_boxes]])

    testing.display(detection_boxes,original)
    print(results[0][1])
