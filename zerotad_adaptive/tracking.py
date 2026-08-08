from __future__ import annotations


import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from ultralytics import __file__ as ultralytics_file
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML

from .common import (
    MotionSnapshot,
    TrackObservation,
    TrackState,
    VEHICLE_CLASS_IDS,
    robust_mad,
    velocity_innovation,
    weighted_median,
)

BYTETRACK_CFG = Path(ultralytics_file).resolve().parent / "cfg" / "trackers" / "bytetrack.yaml"
MAX_NEIGHBORS = 16


def build_byte_tracker(analysis_fps: float) -> BYTETracker:


    args = IterableSimpleNamespace(**YAML.load(BYTETRACK_CFG))

    effective_fps = max(1, int(round(float(analysis_fps))))
    original_buffer = max(1, int(getattr(args, "track_buffer", 30)))


    args.track_buffer = max(
        1,
        int(effective_fps / 30.0 * original_buffer),
    )

    print(
        f"[INFO] ByteTrack: analysis_fps={effective_fps}, "
        f"track_buffer={args.track_buffer} analysis frames"
    )

    return BYTETracker(args)


def assign_track_ids(
    detections: List[Dict[str, Any]],
    tracker: BYTETracker,
    frame_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if detections:
        boxes = np.asarray(
            [
                [
                    *[float(value) for value in detection["bbox_xyxy"]],
                    float(detection["det_conf"]),
                    float(detection["class_id"]),
                ]
                for detection in detections
            ],
            dtype=np.float32,
        )
    else:
        boxes = np.zeros((0, 6), dtype=np.float32)

    tracked = tracker.update(Boxes(boxes, orig_shape=frame_shape))
    if tracked.size == 0:
        return []

    output: List[Dict[str, Any]] = []
    for row in tracked:
        x1, y1, x2, y2, track_id, score, class_id, detection_index = row.tolist()
        index = int(detection_index)
        item = dict(detections[index]) if 0 <= index < len(detections) else {}
        item.update(
            {
                "track_id": int(track_id),
                "class_id": int(class_id),
                "det_conf": float(score),
                "bbox_xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "x_center": float(0.5 * (x1 + x2)),
                "y_center": float(0.5 * (y1 + y2)),
                "width": max(1.0, float(x2 - x1)),
                "height": max(1.0, float(y2 - y1)),
            }
        )
        output.append(item)
    output.sort(key=lambda item: int(item["track_id"]))
    return output


def make_observation(
    frame_id: int,
    bbox: Tuple[float, float, float, float],
    confidence: float,
) -> TrackObservation:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    ground_x = 0.5 * (x1 + x2)
    ground_y = y1 + 0.96 * height
    return TrackObservation(
        frame_id=int(frame_id),
        bbox_xyxy=(x1, y1, x2, y2),
        ground_x=float(ground_x),
        ground_y=float(ground_y),
        scale=float(max(1.0, math.sqrt(width * height))),
        det_conf=float(np.clip(confidence, 0.0, 1.0)),
    )


def update_track_states(
    states: Dict[int, TrackState],
    observations: Sequence[Dict[str, Any]],
    frame_id: int,
    history_size: int,
) -> None:
    for row in observations:
        track_id = int(row["track_id"])
        observation = make_observation(frame_id, row["bbox_xyxy"], float(row["det_conf"]))
        state = states.get(track_id)
        if state is None:
            state = TrackState(
                track_id=track_id,
                class_id=int(row["class_id"]),
                last_frame=int(frame_id),
            )
            states[track_id] = state
        state.class_id = int(row["class_id"])
        state.last_frame = int(frame_id)
        state.observations.append(observation)
        state.seen_hits += 1

        limit = max(6, int(history_size))
        if len(state.observations) > limit:
            state.observations = state.observations[-limit:]


def motion_snapshot(state: TrackState) -> MotionSnapshot:


    observations = state.observations
    vxs: List[float] = []
    vys: List[float] = []
    dts: List[int] = []
    confidences: List[float] = []
    resolutions: List[float] = []

    for previous, current in zip(observations[:-1], observations[1:]):
        dt = max(1, int(current.frame_id - previous.frame_id))
        scale = max(20.0, 0.5 * (previous.scale + current.scale))
        vxs.append((current.ground_x - previous.ground_x) / dt / scale)
        vys.append((current.ground_y - previous.ground_y) / dt / scale)
        dts.append(dt)
        confidences.append(float(math.sqrt(max(0.0, previous.det_conf * current.det_conf))))
        resolutions.append(float(1.0 / (dt * scale)))

    take = min(2, len(vxs))
    if take:
        recent_weights = confidences[-take:]
        vx = weighted_median(vxs[-take:], recent_weights)
        vy = weighted_median(vys[-take:], recent_weights)
    else:
        vx = vy = 0.0

    resolution = float(np.median(resolutions)) if resolutions else 1.0
    innovation = velocity_innovation(vxs, vys)
    return MotionSnapshot(
        vxs=vxs,
        vys=vys,
        dts=dts,
        confidences=confidences,
        resolutions=resolutions,
        vx=float(vx),
        vy=float(vy),
        speed=float(math.hypot(vx, vy)),
        resolution=max(float(resolution), 1e-9),
        innovation=max(float(innovation), 0.0),
    )


def build_neighbor_map(
    observations: Sequence[Dict[str, Any]],
    max_neighbors: int = MAX_NEIGHBORS,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:


    rows = [row for row in observations if int(row.get("class_id", -1)) in VEHICLE_CLASS_IDS]
    if not rows:
        return {}, {}

    ids = np.asarray([int(row["track_id"]) for row in rows], dtype=np.int64)
    centers = np.asarray(
        [[float(row["x_center"]), float(row["y_center"])] for row in rows],
        dtype=np.float32,
    )
    scales = np.asarray(
        [max(20.0, math.sqrt(float(row["width"]) * float(row["height"]))) for row in rows],
        dtype=np.float32,
    )
    differences = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    pair_scale = np.maximum(20.0, 0.5 * (scales[:, None] + scales[None, :]))
    normalized = distances / pair_scale
    np.fill_diagonal(normalized, np.inf)

    neighbor_map: Dict[int, List[int]] = {}
    distance_map: Dict[Tuple[int, int], float] = {}
    count = max(1, int(max_neighbors))
    for row_index, track_id in enumerate(ids.tolist()):
        available = len(ids) - 1
        if available <= 0:
            neighbor_map[int(track_id)] = []
            continue
        k = min(count, available)
        candidate_indices = np.argpartition(normalized[row_index], k - 1)[:k]
        candidate_indices = candidate_indices[np.argsort(normalized[row_index, candidate_indices])]
        neighbor_ids = [int(ids[index]) for index in candidate_indices.tolist()]
        neighbor_map[int(track_id)] = neighbor_ids
        for index in candidate_indices.tolist():
            distance_map[(int(track_id), int(ids[index]))] = float(normalized[row_index, index])
    return neighbor_map, distance_map
