import cv2
import numpy as np
import time


class Generate_Box_By_ViBe(object):
    def __init__(self, threshold=0.2, scale=1):
        self._threshold = threshold
        self._scale = scale

    def resize(self, image):
        return cv2.resize(image, None, None, fx=self._scale, fy=self._scale,
                          interpolation=cv2.INTER_LINEAR)

    def processAndgenerate(self, image):
        # org_image=image.copy()
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        width = image.shape[1]
        height = image.shape[0]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            boxes.append([y / height, x / width, (y + h) / height, (x + w) / width])
            # cv2.rectangle(org_image, (x, y), (x + w, y + h), (255, 255, 255))
        # cv2.imshow('11', org_image)
        return np.array(boxes)

    def filter(self, proposals=np.array([])):
        org_len = len(proposals)
        start = time.time()
        areas = np.zeros([len(proposals)], dtype=float)
        for i in range(len(proposals)):
            for box in self.boxes:
                areas[i] += self.intern_area(proposals[i], box)
            areas[i] /= self._area(proposals[i])
        keep = np.where(areas >= self._threshold)[0]
        end = time.time()
        print(org_len, len(keep), org_len - len(keep), (end - start) * 1000)
        return np.where(areas >= self._threshold)[0]

    def intern_area(self, proposal, box):
        x1 = np.maximum(proposal[0], box[0])
        y1 = np.maximum(proposal[1], box[1])
        x2 = np.minimum(proposal[2], box[2])
        y2 = np.minimum(proposal[3], box[3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        return w * h

    def _area(self, box):
        w = np.maximum(0.0, box[2] - box[0] + 1)
        h = np.maximum(0.0, box[3] - box[1] + 1)
        return w * h


if __name__ == '__main__':
    image = cv2.imread('F:/PostGraduate/Projects/background/video/pre/1.jpg', cv2.IMREAD_GRAYSCALE)
    print(image)
    #gray = np.zeros([300, 300, 1], dtype=np.uint8)
    print(image.dtype)
    generate = Generate_Box_By_ViBe()
    print(generate.processAndgenerate(np.array(image)))
    cv2.waitKey()
