from pathlib import Path
from boxmot_botsort.utils import logger as LOGGER
from boxmot_botsort.utils.checks import RequirementsChecker


class ReidAutoBackend():
    def __init__(
        self,
        weights: str = "",
        device: str = "cpu",
        half: bool = False) -> None:
        """
        Initializes the ReidAutoBackend instance with specified weights, device, and precision mode.

        Args:
            weights (Union[str, List[str]]): Path to the model weights. Can be a string or a list of strings; if a list, the first element is used.
            device (torch.device): The device to run the model on, e.g., CPU or GPU.
            half (bool): Whether to use half precision for model inference.
        """
        # super().__init__()
        self.weights = Path(weights)
        self.device = device
        self.half = half
        self.checker = RequirementsChecker()
        self.model = self.get_backend()

    def get_backend(self):
        """
        Returns an instance of the appropriate backend based on the model type.

        Returns:
            An instance of a backend class corresponding to the detected model type.
        
        Raises:
            SystemExit: If no supported model framework is detected.
        """
        # Mapping of conditions to backend constructors
        match self.weights.suffix:
            case ".pt":
                self.checker.check_packages(("torch", ))
                from boxmot_botsort.appearance.backends.pytorch_backend import PyTorchBackend
                backend_class = PyTorchBackend
            case ".pth":
                self.checker.check_packages(("torch", ))
                from boxmot_botsort.appearance.backends.pytorch_backend import PyTorchBackend
                backend_class = PyTorchBackend
            case ".onnx":
                self.checker.check_packages(("onnxruntime-gpu", ))
                from boxmot_botsort.appearance.backends.onnx_backend import ONNXBackend
                backend_class = ONNXBackend
            case ".xml":
                self.checker.check_packages(("openvino", ))
                from boxmot_botsort.appearance.backends.openvino_backend import OpenVinoBackend
                backend_class = OpenVinoBackend
            case _:
                LOGGER.error("This model framework is not supported yet!")
                exit()
        return backend_class(self.weights)

    def forward(self, im_batch):
        """
        Processes an image batch through the selected backend and returns the processed batch.

        Args:
            im_batch (torch.Tensor): The batch of images to process.

        Returns:
            torch.Tensor: The processed image batch.
        """
        return self.backend.get_features(im_batch)

