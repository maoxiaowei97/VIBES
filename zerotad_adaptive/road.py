from __future__ import annotations


import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .common import (
    EPS,
    MotionSnapshot,
    RoadAxis,
    RoadEstimate,
    TrackState,
    VEHICLE_CLASS_IDS,
    clip01,
    combine_confidences,
    cosine,
    safe_unit,
    sample_reliability,
    equivalent_temporal_sample_count,
    weighted_median,
)


class RoadDirectionField:


    def __init__(self, original_w: int, original_h: int, work_w: int, work_h: int) -> None:
        self.original_w = int(original_w)
        self.original_h = int(original_h)
        self.work_w = int(work_w)
        self.work_h = int(work_h)
        self.scale_x = self.work_w / max(1.0, float(self.original_w))
        self.scale_y = self.work_h / max(1.0, float(self.original_h))

        automatic_cell = int(round(min(self.work_w, self.work_h) / 12.0))
        self.cell = int(np.clip(automatic_cell, 40, 80))
        self.rows = max(1, int(math.ceil(self.work_h / self.cell)))
        self.cols = max(1, int(math.ceil(self.work_w / self.cell)))
        shape = (self.rows, self.cols)
        self.cos2 = np.zeros(shape, dtype=np.float32)
        self.sin2 = np.zeros(shape, dtype=np.float32)
        self.weight = np.zeros(shape, dtype=np.float32)
        self.support = np.zeros(shape, dtype=np.int16)
        self.confidence = np.zeros(shape, dtype=np.float32)


        self.traffic_weight = np.zeros(shape, dtype=np.float32)
        self.traffic_support = np.zeros(shape, dtype=np.float32)


        self.traffic_profile_weight = np.zeros(shape, dtype=np.float32)
        self.traffic_speed_sum = np.zeros(shape, dtype=np.float32)
        self.traffic_speed_sq_sum = np.zeros(shape, dtype=np.float32)
        self.traffic_log_scale_sum = np.zeros(shape, dtype=np.float32)
        self.traffic_log_aspect_sum = np.zeros(shape, dtype=np.float32)

        self.background: Optional[np.ndarray] = None
        self.segments: List[Tuple[float, float, float, float, float]] = []

    @property
    def ready(self) -> bool:
        return bool(np.any(self.weight > 0.0))

    def add_segment(self, x1: float, y1: float, x2: float, y2: float, weight: float) -> None:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(20.0, 0.025 * self.work_w):
            return
        self.segments.append((float(x1), float(y1), float(x2), float(y2), float(weight)))

    def finalize(self) -> None:


        for x1, y1, x2, y2, line_weight in self.segments:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            cosine2, sine2 = math.cos(2.0 * theta), math.sin(2.0 * theta)
            samples = max(2, int(math.ceil(length / max(8.0, 0.45 * self.cell))))
            for t in np.linspace(0.0, 1.0, num=samples):
                x = x1 + t * dx
                y = y1 + t * dy
                column = int(np.clip(x // self.cell, 0, self.cols - 1))
                row = int(np.clip(y // self.cell, 0, self.rows - 1))
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        grid_row, grid_column = row + delta_row, column + delta_column
                        if not (0 <= grid_row < self.rows and 0 <= grid_column < self.cols):
                            continue
                        spatial = 1.0 / (1.0 + delta_row * delta_row + delta_column * delta_column)
                        contribution = float(line_weight * spatial)
                        self.cos2[grid_row, grid_column] += contribution * cosine2
                        self.sin2[grid_row, grid_column] += contribution * sine2
                        self.weight[grid_row, grid_column] += contribution
                        self.support[grid_row, grid_column] += 1

        positive = self.weight[self.weight > 0.0]
        if positive.size == 0:
            return
        reference_weight = max(float(np.median(positive)), EPS)
        resultant = np.sqrt(self.cos2 * self.cos2 + self.sin2 * self.sin2)
        coherence = resultant / np.maximum(self.weight, EPS)
        amount_reliability = self.weight / (self.weight + reference_weight)
        support_reliability = 1.0 - 1.0 / np.sqrt(self.support.astype(np.float32) + 1.0)
        self.confidence = np.clip(
            coherence * np.sqrt(np.maximum(amount_reliability * support_reliability, 0.0)),
            0.0,
            1.0,
        )

    def query(self, x_original: float, y_original: float) -> RoadAxis:
        if not self.ready:
            return RoadAxis()
        x = float(x_original) * self.scale_x
        y = float(y_original) * self.scale_y
        column = int(np.clip(x // self.cell, 0, self.cols - 1))
        row = int(np.clip(y // self.cell, 0, self.rows - 1))

        sum_cosine = sum_sine = sum_weight = 0.0
        local_confidences: List[float] = []
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                grid_row, grid_column = row + delta_row, column + delta_column
                if not (0 <= grid_row < self.rows and 0 <= grid_column < self.cols):
                    continue
                confidence = float(self.confidence[grid_row, grid_column])
                if confidence <= 0.0 or self.weight[grid_row, grid_column] <= 0.0:
                    continue
                distance_weight = 1.0 / (1.0 + delta_row * delta_row + delta_column * delta_column)
                weight = confidence * distance_weight
                sum_cosine += weight * float(
                    self.cos2[grid_row, grid_column] / max(self.weight[grid_row, grid_column], EPS)
                )
                sum_sine += weight * float(
                    self.sin2[grid_row, grid_column] / max(self.weight[grid_row, grid_column], EPS)
                )
                sum_weight += weight
                local_confidences.append(confidence)

        if sum_weight <= EPS:
            return RoadAxis()
        coherence = math.hypot(sum_cosine, sum_sine) / sum_weight
        theta = 0.5 * math.atan2(sum_sine, sum_cosine)
        axis_x, axis_y = math.cos(theta), math.sin(theta)
        confidence = combine_confidences(
            [coherence, float(np.median(local_confidences)), sample_reliability(len(local_confidences))]
        )
        return RoadAxis(float(axis_x), float(axis_y), confidence, "static_lines", True)

    def update_traffic_occupancy(
        self,
        states: Dict[int, TrackState],
        motions: Dict[int, MotionSnapshot],
        active_track_ids: Sequence[int],
        sampling_interval_scale: float = 1.0,
    ) -> None:


        sampling_interval_scale = max(1.0, float(sampling_interval_scale))
        for track_id in active_track_ids:
            state = states.get(int(track_id))
            motion = motions.get(int(track_id))
            if (
                state is None
                or motion is None
                or not motion.ready
                or state.class_id not in VEHICLE_CLASS_IDS
                or not state.observations
            ):
                continue

            recent = state.observations[-min(3, len(state.observations)) :]
            detection_confidence = float(
                np.median([observation.det_conf for observation in recent])
            )
            track_confidence = sample_reliability(
                equivalent_temporal_sample_count(
                    len(motion.vxs), sampling_interval_scale
                )
            )


            excess_motion = max(0.0, motion.speed - motion.resolution)
            motion_signal = excess_motion / max(
                motion.speed + motion.innovation + motion.resolution, EPS
            )
            path_observations = state.observations[-min(5, len(state.observations)) :]
            if len(path_observations) >= 2:
                net_displacement = math.hypot(
                    path_observations[-1].ground_x - path_observations[0].ground_x,
                    path_observations[-1].ground_y - path_observations[0].ground_y,
                )
                path_length = sum(
                    math.hypot(
                        current.ground_x - previous.ground_x,
                        current.ground_y - previous.ground_y,
                    )
                    for previous, current in zip(
                        path_observations[:-1], path_observations[1:]
                    )
                )
                path_coherence = net_displacement / max(path_length, EPS)
            else:
                path_coherence = 0.0


            path_scales = [max(20.0, observation.scale) for observation in path_observations]
            median_path_scale = float(np.median(path_scales)) if path_scales else 20.0
            normalized_translation = net_displacement / max(median_path_scale, EPS)
            translation_support = normalized_translation / (1.0 + normalized_translation)

            contribution = sampling_interval_scale * (
                combine_confidences([detection_confidence, track_confidence])
                * motion_signal
                * motion_signal
                * clip01(path_coherence)
                * clip01(path_coherence)
                * math.sqrt(max(0.0, translation_support))
            )
            if contribution <= EPS:
                continue


            profile_observations = path_observations[-min(5, len(path_observations)) :]
            observation_normalizer = max(1.0, float(len(profile_observations)))
            for observation in profile_observations:
                x = float(observation.ground_x) * self.scale_x
                y = float(observation.ground_y) * self.scale_y
                column = int(np.clip(x // self.cell, 0, self.cols - 1))
                row = int(np.clip(y // self.cell, 0, self.rows - 1))
                x1, y1, x2, y2 = observation.bbox_xyxy
                width = max(1.0, float(x2 - x1))
                height = max(1.0, float(y2 - y1))
                log_scale = math.log(max(1.0, float(observation.scale)))
                log_aspect = math.log(max(width / height, EPS))
                observation_weight = (
                    contribution
                    * clip01(observation.det_conf)
                    / observation_normalizer
                )


                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        grid_row = row + delta_row
                        grid_column = column + delta_column
                        if not (
                            0 <= grid_row < self.rows
                            and 0 <= grid_column < self.cols
                        ):
                            continue
                        spatial = 1.0 / (
                            1.0 + delta_row * delta_row + delta_column * delta_column
                        )
                        weight = float(observation_weight * spatial)
                        self.traffic_weight[grid_row, grid_column] += weight
                        self.traffic_support[grid_row, grid_column] += float(
                            spatial / observation_normalizer
                        )
                        self.traffic_profile_weight[grid_row, grid_column] += weight
                        self.traffic_speed_sum[grid_row, grid_column] += float(
                            weight * motion.speed
                        )
                        self.traffic_speed_sq_sum[grid_row, grid_column] += float(
                            weight * motion.speed * motion.speed
                        )
                        self.traffic_log_scale_sum[grid_row, grid_column] += float(
                            weight * log_scale
                        )
                        self.traffic_log_aspect_sum[grid_row, grid_column] += float(
                            weight * log_aspect
                        )

    def query_traffic_occupancy(
        self, x_original: float, y_original: float
    ) -> float:


        x = float(x_original) * self.scale_x
        y = float(y_original) * self.scale_y
        column = int(np.clip(x // self.cell, 0, self.cols - 1))
        row = int(np.clip(y // self.cell, 0, self.rows - 1))

        evidence = 0.0
        support = 0.0
        normalizer = 0.0
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                grid_row = row + delta_row
                grid_column = column + delta_column
                if not (
                    0 <= grid_row < self.rows
                    and 0 <= grid_column < self.cols
                ):
                    continue
                spatial = 1.0 / (
                    1.0 + delta_row * delta_row + delta_column * delta_column
                )
                evidence += spatial * float(
                    self.traffic_weight[grid_row, grid_column]
                )
                support += spatial * float(
                    self.traffic_support[grid_row, grid_column]
                )
                normalizer += spatial

        if normalizer <= EPS or evidence <= EPS:
            return 0.0
        evidence /= normalizer
        support /= normalizer


        evidence_confidence = 1.0 - math.exp(-max(0.0, evidence))
        support_confidence = 1.0 - 1.0 / math.sqrt(max(0.0, support) + 1.0)


        return clip01(
            1.0
            - (1.0 - evidence_confidence) * (1.0 - support_confidence)
        )

    def query_traffic_profile(
        self,
        x_original: float,
        y_original: float,
        bbox_xyxy: Tuple[float, float, float, float] | None = None,
    ) -> Dict[str, float]:


        x = float(x_original) * self.scale_x
        y = float(y_original) * self.scale_y
        column = int(np.clip(x // self.cell, 0, self.cols - 1))
        row = int(np.clip(y // self.cell, 0, self.rows - 1))

        total_weight = 0.0
        speed_sum = 0.0
        speed_sq_sum = 0.0
        log_scale_sum = 0.0
        log_aspect_sum = 0.0
        local_support = 0.0
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                grid_row = row + delta_row
                grid_column = column + delta_column
                if not (
                    0 <= grid_row < self.rows
                    and 0 <= grid_column < self.cols
                ):
                    continue
                spatial = 1.0 / (
                    1.0 + delta_row * delta_row + delta_column * delta_column
                )
                cell_weight = float(
                    self.traffic_profile_weight[grid_row, grid_column]
                )
                weighted = spatial * cell_weight
                if weighted <= EPS:
                    continue
                total_weight += weighted
                speed_sum += spatial * float(
                    self.traffic_speed_sum[grid_row, grid_column]
                )
                speed_sq_sum += spatial * float(
                    self.traffic_speed_sq_sum[grid_row, grid_column]
                )
                log_scale_sum += spatial * float(
                    self.traffic_log_scale_sum[grid_row, grid_column]
                )
                log_aspect_sum += spatial * float(
                    self.traffic_log_aspect_sum[grid_row, grid_column]
                )
                local_support += spatial

        if total_weight <= EPS:
            return {
                "speed": 0.0,
                "speed_sigma": 0.0,
                "confidence": 0.0,
                "expected_scale": 0.0,
                "expected_aspect": 0.0,
                "geometry_confidence": 0.0,
            }

        speed = speed_sum / total_weight
        speed_variance = max(0.0, speed_sq_sum / total_weight - speed * speed)
        expected_log_scale = log_scale_sum / total_weight
        expected_log_aspect = log_aspect_sum / total_weight

        amount_confidence = 1.0 - math.exp(-max(0.0, total_weight))
        support_confidence = 1.0 - 1.0 / math.sqrt(max(0.0, local_support) + 1.0)
        confidence = clip01(
            1.0
            - (1.0 - amount_confidence) * (1.0 - support_confidence)
        )

        expected_scale = math.exp(expected_log_scale)
        expected_aspect = math.exp(expected_log_aspect)
        geometry_confidence = 0.0
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            candidate_scale = math.sqrt(width * height)
            candidate_aspect = width / height


            scale_ratio = candidate_scale / max(expected_scale, EPS)
            aspect_ratio = candidate_aspect / max(expected_aspect, EPS)
            scale_match = min(scale_ratio, 1.0 / max(scale_ratio, EPS))
            aspect_match = min(aspect_ratio, 1.0 / max(aspect_ratio, EPS))
            geometry_confidence = math.sqrt(
                max(0.0, clip01(scale_match) * clip01(aspect_match))
            )

        return {
            "speed": float(max(0.0, speed)),
            "speed_sigma": float(math.sqrt(speed_variance)),
            "confidence": float(confidence),
            "expected_scale": float(expected_scale),
            "expected_aspect": float(expected_aspect),
            "geometry_confidence": float(geometry_confidence),
        }

    def save_traffic_debug(self, output_dir: Path) -> None:


        if self.background is None:
            return
        positive = self.traffic_weight[self.traffic_weight > 0.0]
        if positive.size == 0:
            return
        scale = max(float(np.quantile(positive, 0.90)), EPS)
        normalized = np.clip(self.traffic_weight / scale, 0.0, 1.0)
        heat_small = (255.0 * normalized).astype(np.uint8)
        heat = cv2.resize(
            heat_small,
            (self.work_w, self.work_h),
            interpolation=cv2.INTER_LINEAR,
        )
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(self.background, 0.65, colored, 0.35, 0.0)
        cv2.imwrite(str(output_dir / "traffic_occupancy_debug.jpg"), overlay)

    def save_debug(self, output_dir: Path) -> None:
        if self.background is None:
            return
        canvas = self.background.copy()
        median_segment_weight = (
            float(np.median([segment[4] for segment in self.segments]))
            if self.segments
            else 0.0
        )
        for x1, y1, x2, y2, weight in self.segments:
            thickness = 1 if weight < median_segment_weight else 2
            cv2.line(
                canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        positive_confidence = self.confidence[self.confidence > 0.0]
        display_threshold = (
            float(np.quantile(positive_confidence, 0.55)) if positive_confidence.size else float("inf")
        )
        for row in range(self.rows):
            for column in range(self.cols):
                confidence = float(self.confidence[row, column])
                if confidence < display_threshold:
                    continue
                theta = 0.5 * math.atan2(float(self.sin2[row, column]), float(self.cos2[row, column]))
                axis_x, axis_y = math.cos(theta), math.sin(theta)
                center_x = int(round((column + 0.5) * self.cell))
                center_y = int(round((row + 0.5) * self.cell))
                length = int(round(0.35 * self.cell * (0.5 + confidence)))
                point1 = (
                    int(round(center_x - length * axis_x)),
                    int(round(center_y - length * axis_y)),
                )
                point2 = (
                    int(round(center_x + length * axis_x)),
                    int(round(center_y + length * axis_y)),
                )
                cv2.line(canvas, point1, point2, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / "road_direction_debug.jpg"), canvas)


def _sample_background(video_path: Path, max_side: int = 1280) -> Tuple[List[np.ndarray], int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0, 0
    try:
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        end = min(total - 1, 240)
        indices = np.linspace(0, end, num=min(8, end + 1), dtype=int)
        frames: List[np.ndarray] = []
        for frame_id in indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            if original_width <= 0 or original_height <= 0:
                original_width, original_height = int(width), int(height)
            ratio = min(1.0, float(max_side) / max(height, width))
            if ratio < 0.999:
                frame = cv2.resize(
                    frame,
                    (max(2, int(round(width * ratio))), max(2, int(round(height * ratio)))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
        return frames, original_width, original_height
    finally:
        cap.release()


def _automatic_canny(gray: np.ndarray) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0.0, 0.67 * median))
    upper = int(min(255.0, 1.33 * median))
    return cv2.Canny(gray, lower, max(lower + 1, upper))


def build_road_direction_field(video_path: Path, output_dir: Path) -> RoadDirectionField:


    frames, original_width, original_height = _sample_background(video_path)
    if not frames:
        field = RoadDirectionField(
            max(1, original_width),
            max(1, original_height),
            max(1, original_width),
            max(1, original_height),
        )
        print("[WARN] 无法建立静态道路方向，将只使用局部车辆轨迹。")
        return field

    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    work_height, work_width = background.shape[:2]
    field = RoadDirectionField(original_width, original_height, work_width, work_height)
    field.background = background

    hls = cv2.cvtColor(background, cv2.COLOR_BGR2HLS)
    hue, lightness, saturation = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    light_threshold = float(np.quantile(lightness, 0.72))
    saturation_threshold = float(np.quantile(saturation, 0.72))
    white = ((lightness >= light_threshold) & (saturation <= saturation_threshold)).astype(np.uint8) * 255
    yellow = (
        (hue >= 10)
        & (hue <= 42)
        & (lightness >= float(np.quantile(lightness, 0.45)))
        & (saturation >= float(np.quantile(saturation, 0.55)))
    ).astype(np.uint8) * 255
    marking = cv2.bitwise_or(white, yellow)
    marking = cv2.morphologyEx(
        marking,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        iterations=1,
    )
    marking_dilated = cv2.dilate(marking, np.ones((3, 3), np.uint8), iterations=1)

    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = _automatic_canny(cv2.GaussianBlur(gray, (5, 5), 0))
    marking_edges = _automatic_canny(marking)
    evidence = cv2.bitwise_or(marking_edges, cv2.bitwise_and(edges, marking_dilated))
    evidence[: int(round(0.10 * work_height)), :] = 0

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(evidence)[0]
    if detected is not None:
        minimum_length = max(22.0, 0.025 * work_width)
        candidates: List[Tuple[float, float, float, float, float]] = []
        for raw in detected.reshape(-1, 4):
            x1, y1, x2, y2 = [float(value) for value in raw.tolist()]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < minimum_length or 0.5 * (y1 + y2) < 0.10 * work_height:
                continue
            sample_count = max(8, int(round(length / 6.0)))
            xs = np.clip(np.linspace(x1, x2, sample_count).round().astype(int), 0, work_width - 1)
            ys = np.clip(np.linspace(y1, y2, sample_count).round().astype(int), 0, work_height - 1)
            marking_support = float(np.mean(marking_dilated[ys, xs] > 0))
            edge_support = float(np.mean(evidence[ys, xs] > 0))
            support = max(marking_support, edge_support)
            if support <= 0.0:
                continue
            weight = float(length * support)
            candidates.append((x1, y1, x2, y2, weight))

        candidates.sort(key=lambda item: item[4], reverse=True)
        selected = candidates[:180]
        median_weight = (
            float(np.median([item[4] for item in selected])) if selected else 1.0
        )
        for x1, y1, x2, y2, raw_weight in selected:

            field.add_segment(x1, y1, x2, y2, raw_weight / max(median_weight, EPS))

    field.finalize()
    field.save_debug(output_dir)
    print(
        f"[INFO] Road direction: {len(frames)} background frames, "
        f"{len(field.segments)} line segments, grid={field.rows}x{field.cols}, ready={field.ready}"
    )
    return field


def _direction_hint(
    state: TrackState,
    motion: MotionSnapshot,
    sampling_interval_scale: float = 1.0,
) -> Tuple[Optional[Tuple[float, float]], float]:


    if not motion.ready:
        return None, 0.0
    if len(motion.vxs) >= 4:
        values_x = motion.vxs[:-2]
        values_y = motion.vys[:-2]
        weights = motion.confidences[:-2]
    else:
        values_x = motion.vxs
        values_y = motion.vys
        weights = motion.confidences
    if not values_x:
        return None, 0.0

    units: List[Tuple[float, float]] = []
    valid_weights: List[float] = []
    for vx, vy, weight in zip(values_x, values_y, weights):
        speed = math.hypot(vx, vy)
        if speed <= EPS:
            continue
        units.append((vx / speed, vy / speed))
        valid_weights.append(max(float(weight), EPS))
    if not units:
        return None, 0.0

    sum_x = sum(unit[0] * weight for unit, weight in zip(units, valid_weights))
    sum_y = sum(unit[1] * weight for unit, weight in zip(units, valid_weights))
    total_weight = sum(valid_weights)
    coherence = math.hypot(sum_x, sum_y) / max(total_weight, EPS)
    signal = motion.speed / max(motion.speed + motion.innovation + motion.resolution, EPS)
    confidence = combine_confidences(
        [
            coherence,
            sample_reliability(
                equivalent_temporal_sample_count(len(units), sampling_interval_scale)
            ),
            signal,
            float(np.median(valid_weights)),
        ]
    )
    return safe_unit(sum_x, sum_y), confidence


def estimate_road_axis(
    track_id: int,
    states: Dict[int, TrackState],
    motions: Dict[int, MotionSnapshot],
    neighbor_ids: Sequence[int],
    neighbor_distances: Dict[Tuple[int, int], float],
    road_field: RoadDirectionField,
    sampling_interval_scale: float = 1.0,
) -> RoadEstimate:


    state = states[track_id]
    motion = motions[track_id]
    latest = state.observations[-1]
    static_axis = road_field.query(latest.ground_x, latest.ground_y)
    own_hint, own_confidence = _direction_hint(
        state, motion, sampling_interval_scale=sampling_interval_scale
    )

    vector_rows: List[Tuple[int, float, float, float]] = []
    for neighbor_id in neighbor_ids:
        neighbor_motion = motions.get(neighbor_id)
        neighbor_state = states.get(neighbor_id)
        if neighbor_motion is None or neighbor_state is None or not neighbor_motion.ready:
            continue
        if neighbor_state.class_id not in VEHICLE_CLASS_IDS:
            continue

        detection_confidence = float(
            np.median([observation.det_conf for observation in neighbor_state.observations[-3:]])
        )
        track_confidence = sample_reliability(
            equivalent_temporal_sample_count(
                len(neighbor_motion.vxs), sampling_interval_scale
            )
        )
        motion_signal = neighbor_motion.speed / max(
            neighbor_motion.speed + neighbor_motion.innovation + neighbor_motion.resolution,
            EPS,
        )
        distance = neighbor_distances.get((track_id, neighbor_id), float("inf"))
        distance_weight = 0.0 if not math.isfinite(distance) else 1.0 / (1.0 + distance)
        alignment_weight = 1.0
        if static_axis.ready:
            alignment_weight = abs(
                cosine((neighbor_motion.vx, neighbor_motion.vy), (static_axis.x, static_axis.y))
            )
        quality = combine_confidences([detection_confidence, track_confidence, motion_signal])
        weight = quality * distance_weight * alignment_weight
        if weight > EPS and neighbor_motion.speed > EPS:
            vector_rows.append((neighbor_id, neighbor_motion.vx, neighbor_motion.vy, weight))

    anchor = own_hint
    if anchor is None and vector_rows:
        strongest = max(vector_rows, key=lambda item: item[3])
        anchor = safe_unit(strongest[1], strongest[2])

    flow_axis: Optional[Tuple[float, float]] = None
    flow_confidence = 0.0
    if vector_rows:
        sum_x = sum_y = total_weight = 0.0
        for _neighbor_id, vx, vy, weight in vector_rows:
            unit_x, unit_y = safe_unit(vx, vy)
            if anchor is not None and cosine((unit_x, unit_y), anchor) < 0.0:
                unit_x, unit_y = -unit_x, -unit_y
            sum_x += unit_x * weight
            sum_y += unit_y * weight
            total_weight += weight
        if total_weight > EPS:
            flow_axis = safe_unit(sum_x, sum_y)
            coherence = math.hypot(sum_x, sum_y) / total_weight
            support_confidence = total_weight / (1.0 + total_weight)
            flow_confidence = combine_confidences([coherence, support_confidence])

    if static_axis.ready:
        axis_x, axis_y = safe_unit(static_axis.x, static_axis.y)
        sign_hint = own_hint or flow_axis
        if sign_hint is not None and cosine((axis_x, axis_y), sign_hint) < 0.0:
            axis_x, axis_y = -axis_x, -axis_y

        if flow_axis is not None:
            flow_x, flow_y = flow_axis
            if cosine((flow_x, flow_y), (axis_x, axis_y)) < 0.0:
                flow_x, flow_y = -flow_x, -flow_y
            static_weight = max(static_axis.confidence, EPS)
            flow_weight = max(flow_confidence, EPS)
            agreement = clip01(abs(cosine((flow_x, flow_y), (axis_x, axis_y))))
            axis_x, axis_y = safe_unit(
                static_weight * axis_x + flow_weight * flow_x,
                static_weight * axis_y + flow_weight * flow_y,
            )
            axis_confidence = combine_confidences(
                [static_axis.confidence, flow_confidence, agreement]
            )
            source = "static+flow"
        else:
            axis_confidence = combine_confidences([static_axis.confidence, own_confidence])
            source = "static"
        axis = RoadAxis(axis_x, axis_y, axis_confidence, source, True)
    elif flow_axis is not None:
        axis = RoadAxis(flow_axis[0], flow_axis[1], flow_confidence, "local_flow", True)
    elif own_hint is not None:
        axis = RoadAxis(own_hint[0], own_hint[1], own_confidence, "self_history", True)
    else:
        return RoadEstimate()

    context_weights: Dict[int, float] = {}
    stop_context_weights: Dict[int, float] = {}
    for neighbor_id, vx, vy, base_weight in vector_rows:
        signed_alignment = cosine((vx, vy), (axis.x, axis.y))
        same_direction = max(0.0, signed_alignment)
        unsigned_direction = abs(signed_alignment)
        if same_direction > 0.0:
            context_weights[int(neighbor_id)] = float(
                base_weight * same_direction
            )
        if unsigned_direction > 0.0:
            stop_context_weights[int(neighbor_id)] = float(
                base_weight * unsigned_direction
            )
    return RoadEstimate(
        axis=axis,
        context_weights=context_weights,
        stop_context_weights=stop_context_weights,
    )
