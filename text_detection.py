
import cv2 as cv
import numpy as np
import math

class CV_Text_Detector():
    def __init__(self,config,source_width,source_height):
        self.detector = cv.dnn.readNet(config)
        self.layer_names = [
                            "feature_fusion/Conv_7/Sigmoid",\
                            "feature_fusion/concat_3"
                           ]
        self._W = source_width
        self._H = source_height

    def process_image(self,image_path):
        image = cv.imread(image_path)
        original = image.copy()

        #dimension ratios
        height, width = image.shape[:2]
        r_height = height/self._H
        r_width = width/self._W

        return image, original, r_height, r_width

    #opencv https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
    def decodeBoundingBoxes(self,scores, geometry, scoreThresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

    def rect_area(self,upper_left,lower_right):
        return (lower_right[0] - upper_left[0])*(lower_right[1] - upper_left[1])

    def box_reduce(self,boxes,confidences):
        """
            creates a single bounding box from overlapping boxes
        """
        def absorb(max_rect,query_rect):
            minx = min(max_rect[0][0],query_rect[0][0])
            miny = min(max_rect[1][0],query_rect[1][0])
            maxx = max(max_rect[0][1],query_rect[0][1])
            maxy = max(max_rect[1][1],query_rect[1][1])
            return ((minx,maxx),(miny,maxy))

        def aggregate_boxes(rectangles):
            max_boxes = []
            threshold = 0.0
            while len(rectangles)!=0:
                max_boxes.append(rectangles.pop(0))
                removal_indicies = []
                i = int(0)
                for ((x1,x2),(y1,y2)) in rectangles:
                    query_area = self.rect_area((x1,y1),(x2,y2))
                    xx1 = max(max_boxes[-1][0][0],x1) 
                    xx2 = min(max_boxes[-1][0][1],x2)
                    yy1 = max(max_boxes[-1][1][0],y1) 
                    yy2 = min(max_boxes[-1][1][1],y2)
                    if ((xx2 - xx1)>0) and ((yy2 - yy1)>0):
                        overlap_area = self.rect_area((xx1,yy1),(xx2,yy2))
                        if overlap_area / query_area > threshold:
                            #absorb
                            new_max = absorb(max_boxes[-1],rectangles[i])
                            max_boxes.pop(-1)
                            max_boxes.append(new_max)
                            removal_indicies.append(i)
                    i += 1
                for index in removal_indicies[::-1]:
                    rectangles.pop(index)
            return max_boxes

        #zip rectangle and confidence score
        _boxes = [(cv.boxPoints(boxes[box]),confidences[box]) for box in range(len(boxes))]

        #sort by confidence score descending
        _boxes.sort(reverse=True,key=lambda x: x[1])

        #retangle box points
        rectangles = []
        for box in _boxes:
            x,y = [],[]
            for point in box[0]:
                x.append(point[0])
                y.append(point[1])
            rectangles.append(((min(x),max(x)),(min(y),max(y))))

        max_boxes = aggregate_boxes(rectangles)

        #group lines of related text;
        # heuristic: that assumes 
        #  if 
        #       boxes are local in vertical space
        #  then
        #       they are parts of a line of related text
        lines = max_boxes
        lines.sort(key=lambda x : x[1][0])
        max_boxes = []
        while len(lines)!=0:
            group = [lines.pop(0)]
            ymin = group[0][1][0]
            ymax = group[0][1][1]
            index = 0
            removal_indicies = []
            for line in lines:
                if line[1][0] > ymax:
                    break
                #if boxes are too far horizontally apart
                if abs(line[0][0] - group[-1][0][1]) > 1500:
                    continue
                group.append(line)
                removal_indicies.append(index)
                index+=1
            for i in removal_indicies[::-1]:
                lines.pop(i)
            xmin = min([l[0][0] for l in group])
            xmax = max([l[0][1] for l in group])
            ymax = max([l[1][1] for l in group])
            max_boxes.append(((xmin,xmax),(ymin,ymax)))

        return aggregate_boxes(max_boxes)

    #return np array of points of the bounding boxes of detected text
    def post_process_detection(self, boxes, confidences, r_width, r_height):

        #expects rect:((xmin,xmax),(ymin,ymax)) -> np.array([(x1,y1),...,(x4,y4),dtype=float32])
        def rectangle_to_box_array(rectangles,r_width,r_height):
            boxes = []
            for ((startX,endX),(startY,endY)) in rectangles:
                # scale the bounding box coordinates based on the respective ratios
                startX = int(startX * r_width)
                startY = int(startY * r_height)
                endX = int(endX * r_width)
                endY = int(endY * r_height)

                boxes.append(
                    np.array([
                        [startX,startY],
                        [startX,endY],
                        [endX,startY],
                        [endX,endY]],
                        dtype='float32'))
            return boxes

        rect_boxes = self.box_reduce(boxes,confidences)

        return rectangle_to_box_array(rect_boxes,r_width,r_height)

    def detect_image(self, image_path):
        image, original, r_height, r_width = self.process_image(image_path)

        blob = cv.dnn.blobFromImage(image,1.0,(self._W,self._H),(123.68, 116.78, 103.94), True, False)
        self.detector.setInput(blob)
        out = self.detector.forward(self.layer_names)
        boxes, confidences = self.decodeBoundingBoxes(out[0],out[1],0.5)
        detection_boxes = self.post_process_detection(boxes,confidences,r_width,r_height)
        return [original,detection_boxes]

    def process_session(self,session):
        pass



