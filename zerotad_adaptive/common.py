from __future__ import annotations


import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-6
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
DEFAULT_CLASS_IDS = "1,2,3,5,7"


@dataclass
class VideoRunResult:
    video_path: Path
    peak_frame: int
    peak_score: float
    event_count: int
    output_dir: Path


@dataclass
class TrackObservation:
    frame_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    ground_x: float
    ground_y: float
    scale: float
    det_conf: float


@dataclass
class TrackState:
    track_id: int
    class_id: int
    last_frame: int
    observations: List[TrackObservation] = field(default_factory=list)
    seen_hits: int = 0


    stop_z_history: List[float] = field(default_factory=list)
    stop_weight_history: List[float] = field(default_factory=list)


    persistent_flow_stop_z_history: List[float] = field(default_factory=list)
    persistent_flow_stop_weight_history: List[float] = field(default_factory=list)


    lateral_signed_z_history: List[float] = field(default_factory=list)
    lateral_weight_history: List[float] = field(default_factory=list)


    impact_posterior_score: float = 0.0


@dataclass
class MotionSnapshot:


    vxs: List[float]
    vys: List[float]
    dts: List[int]
    confidences: List[float]
    resolutions: List[float]
    vx: float
    vy: float
    speed: float
    resolution: float
    innovation: float

    @property
    def ready(self) -> bool:
        return bool(self.vxs)


@dataclass
class RoadAxis:
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0
    source: str = "none"
    ready: bool = False


@dataclass
class RoadEstimate:
    axis: RoadAxis = field(default_factory=RoadAxis)

    context_weights: Dict[int, float] = field(default_factory=dict)


    stop_context_weights: Dict[int, float] = field(default_factory=dict)

    @property
    def context_confidence(self) -> float:
        total = float(sum(max(0.0, value) for value in self.context_weights.values()))
        return float(total / (1.0 + total))

    @property
    def stop_context_confidence(self) -> float:
        total = float(
            sum(max(0.0, value) for value in self.stop_context_weights.values())
        )
        return float(total / (1.0 + total))


@dataclass
class Baseline:
    mean: float
    sigma: float
    confidence: float


@dataclass
class AnomalyResult:
    score: float
    event: str
    lateral_score: float
    overspeed_score: float
    decel_score: float
    stop_score: float
    v_parallel: float
    v_perpendicular: float
    road_axis: RoadAxis
    context_count: int
    confidence: float
    debug: Dict[str, Any]


def clip01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def safe_unit(x: float, y: float) -> Tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= EPS:
        return 1.0, 0.0
    return float(x / norm), float(y / norm)


def cosine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na <= EPS or nb <= EPS:
        return 0.0
    return float((a[0] * b[0] + a[1] * b[1]) / (na * nb))


def sample_reliability(sample_count: int) -> float:


    n = max(0, int(sample_count))
    return float(1.0 - 1.0 / math.sqrt(n + 1.0)) if n > 0 else 0.0


def combine_confidences(values: Sequence[float]) -> float:


    finite = [clip01(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else 0.0


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(arr) & np.isfinite(w) & (w > 0.0)
    arr = arr[valid]
    w = w[valid]
    if arr.size == 0:
        return 0.0
    order = np.argsort(arr)
    arr = arr[order]
    w = w[order]
    cutoff = 0.5 * float(np.sum(w))
    index = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(arr[min(index, arr.size - 1)])


def weighted_mad(
    values: Sequence[float],
    weights: Sequence[float],
    center: float | None = None,
) -> float:
    if not values:
        return 0.0
    location = weighted_median(values, weights) if center is None else float(center)
    deviations = [abs(float(value) - location) for value in values]
    return float(1.4826 * weighted_median(deviations, weights))


def robust_mad(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    median = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - median)))


def velocity_innovation(vxs: Sequence[float], vys: Sequence[float]) -> float:


    if len(vxs) < 2:
        return 0.0
    changes = [
        math.hypot(float(vxs[i] - vxs[i - 1]), float(vys[i] - vys[i - 1]))
        for i in range(1, min(len(vxs), len(vys)))
    ]
    return robust_mad(changes) if len(changes) >= 2 else float(np.median(changes))


def equivalent_temporal_sample_count(
    sample_count: int, sampling_interval_scale: float = 1.0
) -> int:


    n = max(0, int(sample_count))
    scale = max(1.0, float(sampling_interval_scale))
    return int(round(n * scale))


def adaptive_recent_count(
    length: int, sampling_interval_scale: float = 1.0
) -> int:


    n = max(0, int(length))
    if n == 0:
        return 0
    scale = max(1.0, float(sampling_interval_scale))
    equivalent_n = max(1, equivalent_temporal_sample_count(n, scale))
    legacy_recent = max(2, math.ceil(math.sqrt(equivalent_n)))
    current_recent = int(math.ceil(float(legacy_recent) / scale))
    return int(min(n, max(2, current_recent)))
