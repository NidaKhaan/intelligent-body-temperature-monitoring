"""COCO-pretrained torchvision Faster R-CNN, filtered to the 'person' class."""
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)

PERSON_LABEL = 1  # COCO category id for "person" in torchvision's label map

class PersonDetector:
    def __init__(self, variant="mobilenet", score_thr=0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if variant == "mobilenet":
            self.model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        elif variant == "resnet50":
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"unknown variant: {variant}")
        self.model.to(self.device).eval()
        self.score_thr = score_thr

    @torch.inference_mode()
    def detect(self, rgb: np.ndarray):
        """rgb: HxWx3 uint8 (RGB order). Returns (boxes Nx4 xyxy float32, scores N)."""
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8, "expected HxWx3 uint8 RGB"
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model([x])[0]
        keep = (out["labels"] == PERSON_LABEL) & (out["scores"] >= self.score_thr)
        return out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()
