from __future__ import annotations


import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from sahi.auto_model import AutoDetectionModel
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

try:
    from sahi.predict import get_prediction
except ImportError:
    get_prediction = None

from .common import EPS


FAR_ROI_TOP_RATIO = 0.15
FAR_ROI_BOTTOM_RATIO = 0.52
FAR_CONFIDENCE = 0.12
FAR_OVERLAP = 0.12


def _cuda_index(device: str) -> int:
    text = str(device).strip().lower()
    if not text.startswith("cuda") or ":" not in text:
        return 0
    try:
        return max(0, int(text.split(":", 1)[1]))
    except ValueError:
        return 0


class YoloTRTSahiModel(UltralyticsDetectionModel):


    def __init__(self, *args: Any, device_index: int = 0, **kwargs: Any) -> None:
        self._device_index = int(device_index)
        super().__init__(*args, **kwargs)

    def load_model(self) -> None:
        from ultralytics import YOLO

        try:
            self.model = YOLO(self.model_path, task="detect")
            self.device = int(self._device_index)
            names = getattr(self.model, "names", None)
            self.category_mapping = (
                {str(key): str(value) for key, value in names.items()}
                if isinstance(names, dict)
                else {"0": "object"}
            )
        except Exception as exc:
            raise TypeError(f"加载 TensorRT 模型失败 {self.model_path}: {exc}") from exc


def build_detection_model(model_path: Path, conf: float, device: str) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"检测模型不存在: {model_path}")

    print(f"[INFO] Loading detector: {model_path} | device={device}")
    if model_path.suffix.lower() == ".engine":
        return YoloTRTSahiModel(
            model_path=str(model_path),
            confidence_threshold=float(conf),
            device="cuda:0",
            device_index=_cuda_index(device),
        )

    effective_device = str(device).strip() or "cpu"
    if effective_device.startswith("cuda") and not torch.cuda.is_available():
        effective_device = "cpu"
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=float(conf),
        device=effective_device,
    )


def _run_standard_prediction(image: np.ndarray, model: Any) -> Any:
    if get_prediction is None:
        raise RuntimeError("当前 SAHI 版本不包含 get_prediction，请升级 SAHI。")
    try:
        return get_prediction(image, model, verbose=0)
    except TypeError:
        return get_prediction(image, model)


def _resize_for_detection(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return frame, 1.0, 1.0
    ratio = float(max_side) / float(longest)
    new_width = max(2, int(round(width * ratio)))
    new_height = max(2, int(round(height * ratio)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, float(width) / new_width, float(height) / new_height


def _append_predictions(
    prediction_list: Sequence[Any],
    output: List[Dict[str, Any]],
    class_ids_keep: set[int],
    frame_width: int,
    frame_height: int,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> None:
    for obj in prediction_list:
        class_id = int(obj.category.id)
        if class_id not in class_ids_keep:
            continue
        bbox = obj.bbox
        x1 = float(bbox.minx) * x_scale + x_offset
        y1 = float(bbox.miny) * y_scale + y_offset
        x2 = float(bbox.maxx) * x_scale + x_offset
        y2 = float(bbox.maxy) * y_scale + y_offset

        x1 = float(np.clip(x1, 0.0, max(frame_width - 1.0, 0.0)))
        y1 = float(np.clip(y1, 0.0, max(frame_height - 1.0, 0.0)))
        x2 = float(np.clip(x2, x1 + 1.0, max(float(frame_width), x1 + 1.0)))
        y2 = float(np.clip(y2, y1 + 1.0, max(float(frame_height), y1 + 1.0)))
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        output.append(
            {
                "class_id": class_id,
                "det_conf": float(obj.score.value),
                "bbox_xyxy": (x1, y1, x2, y2),
                "x_center": 0.5 * (x1 + x2),
                "y_center": 0.5 * (y1 + y2),
                "width": width,
                "height": height,
            }
        )


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    width, height = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = width * height
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(intersection / max(area_a + area_b - intersection, EPS))


def deduplicate_detections(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:


    if not rows:
        return []
    rows = sorted(rows, key=lambda item: float(item["det_conf"]), reverse=True)
    bucket_size = 64.0
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    kept: List[Dict[str, Any]] = []

    for row in rows:
        center_x, center_y = float(row["x_center"]), float(row["y_center"])
        grid_x, grid_y = int(center_x // bucket_size), int(center_y // bucket_size)
        duplicate = False
        for neighbor_x in range(grid_x - 1, grid_x + 2):
            for neighbor_y in range(grid_y - 1, grid_y + 2):
                for index in buckets.get((neighbor_x, neighbor_y), ()):
                    other = kept[index]
                    center_gap = math.hypot(
                        center_x - float(other["x_center"]),
                        center_y - float(other["y_center"]),
                    )
                    center_gate = 0.55 * min(float(row["width"]), float(other["width"]))
                    if center_gap < center_gate or _bbox_iou(row["bbox_xyxy"], other["bbox_xyxy"]) > 0.65:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            buckets[(grid_x, grid_y)].append(len(kept))
            kept.append(row)
    return kept


def detect_frame(
    frame: np.ndarray,
    model: Any,
    class_ids_keep: set[int],
    global_max_side: int,
    run_far_rescue: bool,
    far_slice_size: int,
) -> List[Dict[str, Any]]:


    frame_height, frame_width = frame.shape[:2]
    raw: List[Dict[str, Any]] = []

    global_frame, scale_x, scale_y = _resize_for_detection(frame, int(global_max_side))
    prediction = _run_standard_prediction(global_frame, model)
    _append_predictions(
        prediction.object_prediction_list,
        raw,
        class_ids_keep,
        frame_width,
        frame_height,
        x_scale=scale_x,
        y_scale=scale_y,
    )

    if run_far_rescue:
        top = int(round(FAR_ROI_TOP_RATIO * frame_height))
        bottom = int(round(FAR_ROI_BOTTOM_RATIO * frame_height))
        if bottom - top >= 64:
            roi = frame[top:bottom, :]
            original_conf = getattr(model, "confidence_threshold", None)
            if original_conf is not None:
                model.confidence_threshold = min(float(original_conf), FAR_CONFIDENCE)
            try:
                sliced = get_sliced_prediction(
                    roi,
                    model,
                    slice_height=max(384, int(far_slice_size)),
                    slice_width=max(384, int(far_slice_size)),
                    overlap_height_ratio=FAR_OVERLAP,
                    overlap_width_ratio=FAR_OVERLAP,
                    perform_standard_pred=False,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    verbose=0,
                )
                _append_predictions(
                    sliced.object_prediction_list,
                    raw,
                    class_ids_keep,
                    frame_width,
                    frame_height,
                    y_offset=float(top),
                )
            finally:
                if original_conf is not None:
                    model.confidence_threshold = original_conf

    return deduplicate_detections(raw)
