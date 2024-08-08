import numpy as np
from openvino.runtime import Core

from boxmot_botsort.utils import logger as LOGGER
from boxmot_botsort.appearance.backends.base_backend import BaseModelBackend


class OpenVinoBackend(BaseModelBackend):

    def __init__(self, weights):
        super().__init__(weights)

    def load_model(self, w):
        LOGGER.info(f"Loading {w} for Openvino inference...")
        core = Core()
        model = core.read_model(w)
        self.session = core.compile_model(model, "CPU")
        self.output_name = self.session.output(0)

    def inference_preprocess(self, x):
        # batch
        crops = np.stack(x, axis=0) / 255.0
        # norm
        crops = (crops - self.mean_array) / self.std_array
        # float32
        crops = crops.astype(np.float32)
        # nchw
        crops = crops.transpose(0, 3, 1, 2)
        return crops
    
    def inference_postprocess(self, features):
        # norm
        features = features / np.linalg.norm(features)
        return features

    def forward(self, im_batch):
        features = self.session(im_batch)[self.output_name]
        return features
