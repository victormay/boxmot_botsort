import torch
import numpy as np

from boxmot_botsort.utils import logger as LOGGER
from boxmot_botsort.appearance.backends.base_backend import BaseModelBackend
from boxmot_botsort.appearance.reid_model_factory import (
    build_model,
    get_nr_classes,
    load_pretrained_weights
)

class PyTorchBackend(BaseModelBackend):

    def __init__(self, weights):
        super().__init__(weights)

    def load_model(self, w):
        LOGGER.info(f"Loading {w} for Openvino inference...")
        # Load a PyTorch model
        self.half = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            self.model_name,
            num_classes=get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=self.device.type == "cuda",
        )
        load_pretrained_weights(self.model, w)
        self.model.to(self.device).eval()
        self.model.half() if self.half else self.model.float()
    
    def inference_preprocess(self, x):
        # batch
        crops = np.stack(x, axis=0) / 255.0
        # norm
        crops = (crops - self.mean_array) / self.std_array
        # float32
        crops = crops.astype(np.float16) if self.half else crops.astype(np.float32)
        # nchw
        crops = crops.transpose(0, 3, 1, 2)
        # tensor
        crops = torch.from_numpy(crops)
        # device
        crops = crops.to(self.device)
        return crops
    
    def inference_postprocess(self, features):
        # norm
        features = features / np.linalg.norm(features)
        return features.astype(np.float32) if self.half else features

    def forward(self, im_batch):
        with torch.no_grad():
            features = self.model(im_batch)
        return features.cpu().numpy()
