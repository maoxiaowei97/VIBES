from __future__ import annotations


import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from .common import AnomalyResult, RoadAxis

_CACHE_MAGIC = "zerotad-crop-analysis-v1"


_CROP_DEBUG_KEYS = (
    "context_track_weights",
    "impact_episode_score",
    "motion_resolution",
    "moving_reference",
    "mu_parallel",
    "normal_lane_change_alternative",
    "persistent_stop_score",
    "position_stability",
    "recent_speed",
    "side_slip_likelihood",
    "sigma_parallel",
    "staticness",
    "stop_semantic_dominant",
    "stop_semantic_score",
    "target_speed_noise",
)


@dataclass(frozen=True)
class CropAnalysisRecord:
    frame_id: int
    observations: Tuple[Dict[str, Any], ...]
    results: Dict[int, AnomalyResult]
    neighbor_map: Dict[int, List[int]]
    distance_map: Dict[Tuple[int, int], float]


def _compact_debug(debug: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in _CROP_DEBUG_KEYS:
        if key not in debug:
            continue
        value = debug[key]
        if key == "context_track_weights" and isinstance(value, dict):
            weights: Dict[int, float] = {}
            for track_id, weight in value.items():
                try:
                    weights[int(track_id)] = float(weight)
                except (TypeError, ValueError):
                    continue
            compact[key] = weights
        elif isinstance(value, bool):
            compact[key] = bool(value)
        elif isinstance(value, int):
            compact[key] = int(value)
        else:
            try:
                compact[key] = float(value)
            except (TypeError, ValueError):
                continue
    return compact


def _compact_result(result: AnomalyResult) -> AnomalyResult:
    axis = result.road_axis
    return AnomalyResult(
        score=float(result.score),
        event=str(result.event),
        lateral_score=float(result.lateral_score),
        overspeed_score=float(result.overspeed_score),
        decel_score=float(result.decel_score),
        stop_score=float(result.stop_score),
        v_parallel=float(result.v_parallel),
        v_perpendicular=float(result.v_perpendicular),
        road_axis=RoadAxis(
            x=float(axis.x),
            y=float(axis.y),
            confidence=float(axis.confidence),
            source=str(axis.source),
            ready=bool(axis.ready),
        ),
        context_count=int(result.context_count),
        confidence=float(result.confidence),
        debug=_compact_debug(result.debug),
    )


def _compact_observation(row: Dict[str, Any]) -> Dict[str, Any]:
    bbox = tuple(float(value) for value in row["bbox_xyxy"])
    x1, y1, x2, y2 = bbox
    return {
        "track_id": int(row["track_id"]),
        "class_id": int(row.get("class_id", -1)),
        "det_conf": float(row.get("det_conf", 0.0)),
        "bbox_xyxy": bbox,
        "x_center": float(row.get("x_center", 0.5 * (x1 + x2))),
        "y_center": float(row.get("y_center", 0.5 * (y1 + y2))),
        "width": float(row.get("width", max(0.0, x2 - x1))),
        "height": float(row.get("height", max(0.0, y2 - y1))),
    }


def make_crop_record(
    frame_id: int,
    observations: Sequence[Dict[str, Any]],
    results: Dict[int, AnomalyResult],
    neighbor_map: Dict[int, List[int]],
    distance_map: Dict[Tuple[int, int], float],
) -> CropAnalysisRecord:
    compact_observations = tuple(_compact_observation(row) for row in observations)
    compact_results = {
        int(track_id): _compact_result(result)
        for track_id, result in results.items()
    }
    compact_neighbors = {
        int(track_id): [int(neighbor_id) for neighbor_id in neighbors]
        for track_id, neighbors in neighbor_map.items()
    }


    required_pairs = {
        (int(track_id), int(neighbor_id))
        for track_id, neighbors in compact_neighbors.items()
        for neighbor_id in neighbors
    }
    compact_distances: Dict[Tuple[int, int], float] = {}
    for pair in required_pairs:
        if pair in distance_map:
            compact_distances[pair] = float(distance_map[pair])
            continue
        reverse = (pair[1], pair[0])
        if reverse in distance_map:
            compact_distances[pair] = float(distance_map[reverse])

    return CropAnalysisRecord(
        frame_id=int(frame_id),
        observations=compact_observations,
        results=compact_results,
        neighbor_map=compact_neighbors,
        distance_map=compact_distances,
    )


class CropAnalysisCacheWriter:


    def __init__(
        self,
        path: Path,
        *,
        video_path: Path,
        fps: float,
        frame_stride: int,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("wb", buffering=8 * 1024 * 1024)
        self.record_count = 0
        pickle.dump(
            {
                "magic": _CACHE_MAGIC,
                "video_path": str(Path(video_path).resolve()),
                "fps": float(fps),
                "frame_stride": int(frame_stride),
            },
            self._handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def add(
        self,
        frame_id: int,
        observations: Sequence[Dict[str, Any]],
        results: Dict[int, AnomalyResult],
        neighbor_map: Dict[int, List[int]],
        distance_map: Dict[Tuple[int, int], float],
    ) -> None:
        record = make_crop_record(
            frame_id,
            observations,
            results,
            neighbor_map,
            distance_map,
        )
        pickle.dump(record, self._handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.record_count += 1

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        self._handle.close()
        self._handle = None


def iter_crop_records(path: Path) -> Iterator[CropAnalysisRecord]:
    with Path(path).open("rb", buffering=8 * 1024 * 1024) as handle:
        header = pickle.load(handle)
        if not isinstance(header, dict) or header.get("magic") != _CACHE_MAGIC:
            raise RuntimeError(f"无效的 Crop 分析缓存: {path}")
        while True:
            try:
                record = pickle.load(handle)
            except EOFError:
                break
            if not isinstance(record, CropAnalysisRecord):
                raise RuntimeError(f"Crop 分析缓存记录类型错误: {path}")
            yield record
