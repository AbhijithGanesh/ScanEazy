import cv2
from api.augmentors.manager import Processor


class GaussianBlur(Processor):
    def __init__(self, options, _args):
        self.kSize = tuple(int(x) for x in options.get("kSize", (3, 3)))
        self.sigmaX = int(options.get("sigmaX", 0))

    def apply_filter(self, image, _args):
        return cv2.GaussianBlur(image, self.kSize, self.sigmaX)
