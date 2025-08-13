# SAM2: pip install -e . inside the facebookresearch/sam2 repo
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
import types

# --- SAM2 imports (official) ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import inspect
# --------------------------------

class SAM2Segmentor:
    """
    Image-only segmentor using SAM2. Returns dict with:
      - masks: (N, H, W) float tensor in {0,1}
      - boxes: (N, 4) float tensor [x1,y1,x2,y2]
    Matches the shape contract used by your SAM utilities.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_file: str,                    # e.g. "configs/sam2.1/sam2.1_hiera_l.yaml" or "sam2.1_hiera_l.yaml"
        device: str = "cuda",
        segmentor_width_size: Optional[int] = None,
        apply_postprocessing: bool = False,
        model_cfg: Any = None,               # accepted for hydra compatibility, unused
        **maskgen_kwargs,                    # forwarded to SAM2AutomaticMaskGenerator
    ):
        self.device = device
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device

        # Build core model
        sam2_model = build_sam2(config_file, checkpoint_path, apply_postprocessing=apply_postprocessing)

        # Filter kwargs to what SAM2AutomaticMaskGenerator accepts
        allowed = set(inspect.signature(SAM2AutomaticMaskGenerator.__init__).parameters.keys()) - {"self"}
        maskgen_args = {k: v for k, v in maskgen_kwargs.items() if k in allowed}

        # Create AMG and a predictor shim compatible with our device-moving code
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model, **maskgen_args)
        self.predictor = types.SimpleNamespace(model=sam2_model)  # used by run_inference_custom.py
        self._prompt_predictor = SAM2ImagePredictor(sam2_model)   # optional promptable API

    # ---------- resize utils (mirror your SAM helpers) ----------
    def _preprocess_resize(self, image_np: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Resize to (segmentor_width_size, height) keeping aspect ratio."""
        H, W = image_np.shape[:2]
        if self.segmentor_width_size is None:
            return image_np, (H, W), 1.0
        new_W = int(self.segmentor_width_size)
        new_H = int(round(new_W * H / W))
        resized = cv2.resize(image_np, (new_W, new_H))
        scale = W / float(new_W)  # for boxes rescale back
        return resized, (H, W), scale

    def postprocess_resize(
        self, detections: Dict[str, torch.Tensor], orig_size: Tuple[int, int], update_boxes: bool = True
    ) -> Dict[str, torch.Tensor]:
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        if update_boxes and "boxes" in detections:
            detections["boxes"][:, [0, 2]] = torch.clamp(detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1)
            detections["boxes"][:, [1, 3]] = torch.clamp(detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1)
        return detections
    # ------------------------------------------------------------

    @torch.inference_mode()
    def generate_masks(self, image_bgr: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Automatic mask generation over the whole image (like SAM's SamAutomaticMaskGenerator).
        Input: OpenCV BGR or RGB np.ndarray (H,W,3). SAM2 examples use RGB.
        """
        image_rgb = image_bgr[..., ::-1] if image_bgr.shape[-1] == 3 else image_bgr  # assume BGR->RGB

        # Optional resize for speed
        if self.segmentor_width_size is not None:
            img_resized, orig_size, scale = self._preprocess_resize(image_rgb)
        else:
            img_resized, orig_size, scale = image_rgb, image_rgb.shape[:2], 1.0

        # Generate list[dict] like SAM: each item has 'segmentation' (H,W bool/uint8) and 'bbox' [x,y,w,h]
        preds: List[Dict[str, Any]] = self.mask_generator.generate(img_resized)

        # Convert to tensors (N,H,W) and (N,4 xyxy). If no preds, return empty tensors.
        if len(preds) == 0:
            H, W = orig_size
            device = torch.device(self.device)
            return {
                "masks": torch.zeros((0, H, W), dtype=torch.float32, device=device),
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            }

        masks_np = [p["segmentation"].astype(np.float32) for p in preds]           # (h',w')
        boxes_xywh = np.array([p["bbox"] for p in preds], dtype=np.float32)        # [x,y,w,h] in resized coords

        # Scale boxes back to original width if we resized
        if self.segmentor_width_size is not None:
            boxes_xywh[:, [0, 2]] *= scale  # x,w
            boxes_xywh[:, [1, 3]] *= scale  # y,h

        # xywh -> xyxy
        boxes_xyxy = boxes_xywh.copy()
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]

        # Stack to tensors and (if resized) upsample masks back to original H,W
        device = torch.device(self.device)
        masks = torch.from_numpy(np.stack(masks_np, axis=0)).to(device)            # (N,h',w')
        boxes = torch.from_numpy(boxes_xyxy).to(device)

        # If we resized for inference, upsample the masks to original size
        if self.segmentor_width_size is not None:
            masks = F.interpolate(
                masks.unsqueeze(1), size=orig_size, mode="bilinear", align_corners=False
            )[:, 0, :, :]

        # Clamp boxes to image bounds
        H, W = orig_size
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, W - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, H - 1)

        # Binarize to {0,1} if desired (keep float for later ops)
        masks = (masks > 0.5).float()
        return {"masks": masks, "boxes": boxes}

    # Optional promptable API (clicks/boxes)
    @torch.inference_mode()
    def predict_with_prompts(
        self,
        image_bgr: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box_xyxy: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        image_rgb = image_bgr[..., ::-1]
        self._prompt_predictor.set_image(image_rgb)
        masks, scores, _ = self._prompt_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            multimask_output=multimask_output,
        )
        return {
            "masks": torch.from_numpy(masks.astype(np.float32)).to(self.device)[:, 0],
            "scores": torch.from_numpy(scores.astype(np.float32)).to(self.device),
        }
