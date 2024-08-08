import cv2
import numpy as np
from abc import ABC, abstractmethod


__model_types = [
    "resnet50",
    "resnet101",
    "mlfn",
    "hacnn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "lmbn_n",
    "clip",
]


def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None


class BaseModelBackend(ABC):
    def __init__(self, weights):
        self.weights = weights
        self.model_name = get_model_name(self.weights)
        self.mean_array = np.array([0.485, 0.456, 0.406])
        self.std_array = np.array([0.229, 0.224, 0.225])
        self.resize_dims = (128, 256, 3)
        self.load_model(self.weights)

    def get_crops(self, xyxys, img):
        crops = []
        h, w = img.shape[:2]
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            # resize
            crop = cv2.resize(
                crop,
                self.resize_dims[:2],  # from (x, y) to (128, 256) | (w, h)
            )

            # (cv2) BGR 2 (PIL) RGB. The ReID models have been trained with this channel order
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
        return crops

    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        return features

    def warmup(self):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *self.resize_dims, dtype=np.uint8)
            crops = self.get_crops(xyxys=np.array(
                [[0, 0, 64, 64], [0, 0, 128, 128]]),
                img=im
            )
            crops = self.inference_preprocess(crops)
            self.forward(crops)  # warmup

    @abstractmethod
    def inference_preprocess(self, x):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def inference_postprocess(self, features):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def forward(self, im_batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def load_model(self, w):
        raise NotImplementedError("This method should be implemented by subclasses.")