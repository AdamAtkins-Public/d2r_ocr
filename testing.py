import time
import cv2 as cv
from matplotlib import pyplot as plt

class Timer(object):
    def __enter__(self):
        self.start = time.time()
    def __exit__(self,value,type,traceback):
        self.end = time.time()
    def duration(self):
        return self.end - self.start

def display(boxes,image):
    """
        draws detected rectangles on original image

        boxes contains np.array of scaled rectangle coordinates
            [[startX,startY],[startX,endY],[endX,startY],[endX,endY]]
    """
    #display
    for array in boxes:
        startX, startY = int(array[0][0]), int(array[0][1])
        endX, endY = int(array[3][0]), int(array[3][1])
        # draw the bounding box on the image
        cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output image
    plt.imshow(image)
    plt.show()



