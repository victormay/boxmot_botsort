import numpy as np
import onnxruntime

from boxmot_botsort.utils import logger as LOGGER
from boxmot_botsort.appearance.backends.base_backend import BaseModelBackend


class ONNXBackend(BaseModelBackend):

    def __init__(self, weights):
        super().__init__(weights)

    def load_model(self, w):
        LOGGER.info(f"Loading {w} for OnnxRuntime inference...")
        # TODO gpu 支持
        # providers = (["CUDAExecutionProvider", "CPUExecutionProvider"])
        providers = (["CPUExecutionProvider"])
        self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        input_shape = [i.shape for i in self.session.get_inputs()][0][1:][::-1]
        self.resize_dims = input_shape
        # TODO half 支持
        self.half = False

    def inference_preprocess(self, x):
        # batch
        crops = np.stack(x, axis=0) / 255.0
        # norm
        crops = (crops - self.mean_array) / self.std_array
        # half or float32
        crops = crops.astype(np.float16) if self.half else crops.astype(np.float32)
        # nchw
        crops = crops.transpose(0, 3, 1, 2)
        return crops
    
    def inference_postprocess(self, features):
        # norm
        features = features / np.linalg.norm(features)
        return features

    def forward(self, im_batch):
        features = self.session.run(
            [self.session.get_outputs()[0].name],
            {self.session.get_inputs()[0].name: im_batch},
        )[0]
        return features
