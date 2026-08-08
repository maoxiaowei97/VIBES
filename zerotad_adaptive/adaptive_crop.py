from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .common import AnomalyResult, EPS, clip01, safe_unit
from .config import ensure_dir

_JPEG_QUALITY = 90
_MAX_KEYFRAMES = 6
_MIN_KEYFRAMES = 3

_STOP_LOOKBACK_SECONDS = 12.0
_STOP_AUDIT_SAMPLE_SECONDS = 1.0
_MAX_STOP_VLM_KEYFRAMES = 7
_MAX_PERSISTENT_VLM_KEYFRAMES = 8

_MAX_STOP_ALARM_KEYFRAMES = 6
_MAX_STOP_CAUSE_KEYFRAMES = 10
_STOP_CAUSE_SAMPLE_SECONDS = 0.75
_MAX_FORWARD_OUTCOME_KEYFRAMES = 6
_FORWARD_OUTCOME_SECONDS = 2.4


@dataclass
class CropFramePacket:
    frame_id: int
    time_sec: float
    jpeg_bytes: bytes
    frame_width: int
    frame_height: int
    observations: Dict[int, Dict[str, Any]]
    results: Dict[int, AnomalyResult]
    neighbor_map: Dict[int, List[int]]
    distance_map: Dict[Tuple[int, int], float]

    def decode(self) -> np.ndarray:
        array = np.frombuffer(self.jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法解码 crop 缓存帧: frame={self.frame_id}")
        return frame


@dataclass
class CompletedSegment:
    index: int
    packets: List[CropFramePacket]


@dataclass
class AlarmSample:
    packet_index: int
    track_id: int
    score: float
    event: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    scale: float
    time_sec: float = 0.0


@dataclass
class RegionalProposal:
    representative_track_id: int
    member_track_ids: List[int]

    focus_track_ids: List[int]
    track_relevance: Dict[int, float]
    alarm_packet_indices: List[int]
    alarm_samples: List[AlarmSample]
    event: str
    peak_score: float
    peak_packet_index: int
    core_roi_xyxy: Tuple[float, float, float, float]
    core_center_xy: Tuple[float, float]
    core_scale: float
    effective_alarm_count: float
    recurrence_reliability: float
    spatial_concentration: float
    mean_confidence: float
    credibility: float
    quality: float


@dataclass
class AdaptiveEvent:
    representative_track_id: int
    member_track_ids: List[int]
    focus_track_ids: List[int]
    event: str
    peak_score: float
    peak_packet_index: int
    span_start_index: int
    span_end_index: int
    lookback_seconds: float
    future_seconds: float
    keyframe_indices: List[int]
    keyframe_roles: Dict[int, str]
    phase_indices: Dict[str, int]
    neighbor_ids: List[int]
    neighbor_weights: Dict[int, float]

    roi_xyxy: Tuple[int, int, int, int]
    core_roi_xyxy: Tuple[float, float, float, float]
    core_center_xy: Tuple[float, float]
    core_scale: float
    proposal_quality: float
    proposal_credibility: float
    effective_alarm_count: float
    alarm_frame_count: int
    stop_retrospective: bool
    stop_alarm_focus: bool
    stop_cause_search: bool
    persistent_alarm_focus: bool
    forward_outcome_search: bool
    temporal_mode: str


@dataclass
class RecentEmission:
    roi_xyxy: Tuple[int, int, int, int]
    core_roi_xyxy: Tuple[float, float, float, float]
    core_center_xy: Tuple[float, float]
    core_scale: float
    start_time_sec: float
    end_time_sec: float
    peak_time_sec: float
    quality: float
    temporal_mode: str


@dataclass(frozen=True)
class EventCropArtifact:

    segment_index: int
    segment_start_sec: float
    segment_end_sec: float
    event_index: int
    representative_track_id: int
    member_track_ids: Tuple[int, ...]
    event: str
    peak_score: float
    peak_time_sec: float
    adaptive_start_sec: float
    adaptive_end_sec: float
    metadata_path: Path
    clean_canvas_path: Path
    debug_canvas_path: Path
    keyframe_clean_paths: Tuple[Path, ...]
    keyframe_debug_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class SegmentCropBatch:

    segment_index: int
    segment_start_sec: float
    segment_end_sec: float
    segment_dir: Path
    events: Tuple[EventCropArtifact, ...]


class AdaptiveCropWriter:


    def __init__(
        self,
        output_dir: Path,
        fps: float,
        interval_seconds: float,
        anomaly_threshold: float,
        canvas_max_side: int,
        max_events_per_segment: int = 4,
        focus_base_seconds: float = 4.0,
        enabled: bool = True,
        on_segment_ready: Optional[Callable[[SegmentCropBatch], None]] = None,
    ) -> None:
        self.enabled = bool(enabled and interval_seconds > 0.0)
        self.fps = max(1.0, float(fps))
        self.interval_seconds = max(0.25, float(interval_seconds))
        self.threshold = float(anomaly_threshold)
        self.canvas_max_side = max(720, int(canvas_max_side) if canvas_max_side > 0 else 1280)
        self.max_events_per_segment = min(4, max(1, int(max_events_per_segment)))


        self.focus_base_seconds = max(0.25, float(focus_base_seconds))
        self.stop_lookback_seconds = float(_STOP_LOOKBACK_SECONDS)
        self.history_segment_count = max(
            1, int(math.ceil(self.stop_lookback_seconds / self.interval_seconds))
        )
        self.root = Path(output_dir) / "adaptive_spatiotemporal_crops"
        self.on_segment_ready = on_segment_ready

        self.current_segment_index: Optional[int] = None
        self.current_packets: List[CropFramePacket] = []
        self.pending_segment: Optional[CompletedSegment] = None
        self.history_segments: deque[CompletedSegment] = deque(
            maxlen=self.history_segment_count
        )
        self.recent_emissions: List[RecentEmission] = []
        self.event_serial = 0


        self._closed = False
        self._aborted = False

        self.manifest_handle = None
        self.manifest_writer = None
        if self.enabled:
            ensure_dir(self.root)
            manifest_path = self.root / "crop_manifest.csv"
            self.manifest_handle = manifest_path.open("w", newline="", encoding="utf-8")
            self.manifest_writer = csv.DictWriter(
                self.manifest_handle,
                fieldnames=[
                    "segment_index",
                    "segment_start_sec",
                    "segment_end_sec",
                    "event_index",
                    "representative_track_id",
                    "member_track_ids",
                    "event",
                    "peak_score",
                    "proposal_quality",
                    "proposal_credibility",
                    "effective_alarm_count",
                    "alarm_frame_count",
                    "temporal_mode",
                    "stop_retrospective",
                    "stop_alarm_focus",
                    "stop_cause_search",
                    "forward_outcome_search",
                    "peak_time_sec",
                    "adaptive_start_sec",
                    "adaptive_end_sec",
                    "lookback_seconds",
                    "future_seconds",
                    "keyframe_count",
                    "neighbor_ids",
                    "clean_canvas",
                    "debug_canvas",
                    "metadata_json",
                ],
            )
            self.manifest_writer.writeheader()

    def add_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        observations: Sequence[Dict[str, Any]],
        results: Dict[int, AnomalyResult],
        neighbor_map: Dict[int, List[int]],
        distance_map: Dict[Tuple[int, int], float],
    ) -> None:
        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("AdaptiveCropWriter 已关闭，不能继续写入帧")

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
        )
        if not success:
            return

        observation_map: Dict[int, Dict[str, Any]] = {}
        for row in observations:
            track_id = int(row["track_id"])
            observation_map[track_id] = {
                "track_id": track_id,
                "class_id": int(row.get("class_id", -1)),
                "det_conf": float(row.get("det_conf", 0.0)),
                "bbox_xyxy": tuple(float(v) for v in row["bbox_xyxy"]),
                "x_center": float(row.get("x_center", 0.0)),
                "y_center": float(row.get("y_center", 0.0)),
                "width": float(row.get("width", 0.0)),
                "height": float(row.get("height", 0.0)),
            }

        time_sec = float(frame_id) / self.fps
        packet = CropFramePacket(
            frame_id=int(frame_id),
            time_sec=time_sec,
            jpeg_bytes=bytes(encoded),
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            observations=observation_map,
            results=dict(results),
            neighbor_map={int(k): [int(v) for v in values] for k, values in neighbor_map.items()},
            distance_map={(int(a), int(b)): float(v) for (a, b), v in distance_map.items()},
        )

        segment_index = int(math.floor(time_sec / self.interval_seconds))
        if self.current_segment_index is None:
            self.current_segment_index = segment_index
        elif segment_index != self.current_segment_index:
            self._seal_current_segment()
            self.current_segment_index = segment_index
            self.current_packets = []

        self.current_packets.append(packet)

    def close(self) -> None:


        if not self.enabled or self._closed:
            return
        self._closed = True
        try:
            if not self._aborted:
                self._seal_current_segment()

                if self.pending_segment is not None:
                    self._finalize_pending(future_segment=None)
                    self.history_segments.append(self.pending_segment)
                    self.pending_segment = None
        finally:
            self._close_manifest()

    def abort(self) -> None:


        if not self.enabled or self._closed:
            return
        self._aborted = True
        self._closed = True
        self.current_packets = []
        self.pending_segment = None
        self.history_segments.clear()
        self._close_manifest()

    def _close_manifest(self) -> None:
        if self.manifest_handle is not None:
            self.manifest_handle.flush()
            self.manifest_handle.close()
            self.manifest_handle = None
            self.manifest_writer = None

    def _seal_current_segment(self) -> None:
        if self.current_segment_index is None or not self.current_packets:
            return
        completed = CompletedSegment(
            index=int(self.current_segment_index),
            packets=list(self.current_packets),
        )
        if self.pending_segment is not None:

            self._finalize_pending(future_segment=completed)
            self.history_segments.append(self.pending_segment)
        self.pending_segment = completed
        self.current_packets = []


    def _finalize_pending(self, future_segment: Optional[CompletedSegment]) -> None:
        target = self.pending_segment
        if target is None or not target.packets:
            return

        timeline: List[CropFramePacket] = []
        for segment in self.history_segments:
            timeline.extend(segment.packets)
        target_start = len(timeline)
        timeline.extend(target.packets)
        target_end = len(timeline)
        if future_segment is not None:
            timeline.extend(future_segment.packets)

        proposals = self._build_regional_proposals(
            timeline,
            target_start=target_start,
            target_end=target_end,
        )
        selected_proposals = self._select_top_proposals(proposals)
        events: List[AdaptiveEvent] = []
        for proposal in selected_proposals:
            for event in self._materialize_events(timeline, proposal):
                if self._is_recent_duplicate(timeline, event):
                    continue


                if (
                    not event.temporal_mode.startswith("stop_")
                    and event.alarm_frame_count <= 1
                    and event.peak_score < 1.60 * self.threshold
                    and event.proposal_credibility < 0.58
                ):
                    continue
                events.append(event)


        events.sort(key=_event_vlm_priority, reverse=True)
        events = events[: self.max_events_per_segment]

        segment_index = int(target.index)
        segment_start = segment_index * self.interval_seconds
        segment_end = segment_start + self.interval_seconds
        if events:
            segment_dir = self.root / (
                f"segment_{segment_index:04d}_"
                f"{int(round(1000 * segment_start)):08d}ms_"
                f"{int(round(1000 * segment_end)):08d}ms"
            )
            ensure_dir(segment_dir)
            debug_canvases: List[np.ndarray] = []
            artifacts: List[EventCropArtifact] = []
            for local_index, event in enumerate(events, start=1):
                paths, debug_canvas, artifact = self._write_event(
                    timeline,
                    event,
                    segment_dir,
                    segment_index,
                    segment_start,
                    segment_end,
                    local_index,
                )
                debug_canvases.append(debug_canvas)
                artifacts.append(artifact)
                self._remember_emission(timeline, event)
                if self.manifest_writer is not None:
                    peak_packet = timeline[event.peak_packet_index]
                    start_packet = timeline[event.span_start_index]
                    end_packet = timeline[event.span_end_index]
                    self.manifest_writer.writerow(
                        {
                            "segment_index": segment_index,
                            "segment_start_sec": f"{segment_start:.3f}",
                            "segment_end_sec": f"{segment_end:.3f}",
                            "event_index": self.event_serial,
                            "representative_track_id": event.representative_track_id,
                            "member_track_ids": "|".join(str(v) for v in event.member_track_ids),
                            "event": event.event,
                            "peak_score": f"{event.peak_score:.6f}",
                            "proposal_quality": f"{event.proposal_quality:.6f}",
                            "proposal_credibility": f"{event.proposal_credibility:.6f}",
                            "effective_alarm_count": f"{event.effective_alarm_count:.3f}",
                            "alarm_frame_count": event.alarm_frame_count,
                            "temporal_mode": event.temporal_mode,
                            "stop_retrospective": int(event.stop_retrospective),
                            "stop_alarm_focus": int(event.stop_alarm_focus),
                            "stop_cause_search": int(event.stop_cause_search),
                            "forward_outcome_search": int(event.forward_outcome_search),
                            "peak_time_sec": f"{peak_packet.time_sec:.3f}",
                            "adaptive_start_sec": f"{start_packet.time_sec:.3f}",
                            "adaptive_end_sec": f"{end_packet.time_sec:.3f}",
                            "lookback_seconds": f"{event.lookback_seconds:.3f}",
                            "future_seconds": f"{event.future_seconds:.3f}",
                            "keyframe_count": len(event.keyframe_indices),
                            "neighbor_ids": "|".join(str(v) for v in event.neighbor_ids),
                            "clean_canvas": str(paths["clean"].relative_to(self.root)),
                            "debug_canvas": str(paths["debug"].relative_to(self.root)),
                            "metadata_json": str(paths["json"].relative_to(self.root)),
                        }
                    )

            summary = _compose_image_grid(
                debug_canvases,
                max_side=self.canvas_max_side,
                title=(
                    f"segment {segment_index:04d}  "
                    f"{segment_start:.2f}s - {segment_end:.2f}s  "
                    f"selected={len(events)} / proposals={len(proposals)}  "
                    f"cap={self.max_events_per_segment}"
                ),
                background=(20, 20, 20),
            )
            cv2.imwrite(str(segment_dir / "segment_summary.jpg"), summary)
            if artifacts and self.on_segment_ready is not None:
                batch = SegmentCropBatch(
                    segment_index=segment_index,
                    segment_start_sec=segment_start,
                    segment_end_sec=segment_end,
                    segment_dir=segment_dir.resolve(),
                    events=tuple(artifacts),
                )
                try:
                    self.on_segment_ready(batch)
                except Exception as exc:

                    print(f"[WARN] segment {segment_index:04d} 下游回调失败: {exc}")


        latest_time = timeline[-1].time_sec if timeline else segment_end
        self.recent_emissions = [
            item
            for item in self.recent_emissions
            if latest_time - item.end_time_sec <= 3.0 * self.focus_base_seconds
        ]

    def _build_regional_proposals(
        self,
        timeline: Sequence[CropFramePacket],
        target_start: int,
        target_end: int,
    ) -> List[RegionalProposal]:
        alarms: List[AlarmSample] = []
        for packet_index in range(target_start, target_end):
            packet = timeline[packet_index]
            for track_id, result in packet.results.items():
                observation = packet.observations.get(track_id)
                if observation is None or float(result.score) < self.threshold:
                    continue
                bbox = tuple(float(v) for v in observation["bbox_xyxy"])
                x1, y1, x2, y2 = bbox
                alarms.append(
                    AlarmSample(
                        packet_index=int(packet_index),
                        track_id=int(track_id),
                        score=float(result.score),
                        event=str(result.event),
                        confidence=float(result.confidence),
                        bbox_xyxy=bbox,
                        center_xy=(0.5 * (x1 + x2), 0.5 * (y1 + y2)),
                        scale=math.sqrt(max(1.0, (x2 - x1) * (y2 - y1))),
                        time_sec=float(packet.time_sec),
                    )
                )
        if not alarms:
            return []


        parent = list(range(len(alarms)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for i in range(len(alarms)):
            for j in range(i + 1, len(alarms)):


                temporal_close = abs(alarms[i].time_sec - alarms[j].time_sec) <= (
                    self.focus_base_seconds + EPS
                )
                if not temporal_close:
                    continue
                if alarms[i].track_id == alarms[j].track_id:
                    union(i, j)
                    continue
                if _alarm_samples_share_region(alarms[i], alarms[j]):
                    union(i, j)

        groups: Dict[int, List[AlarmSample]] = {}
        for index, sample in enumerate(alarms):
            groups.setdefault(find(index), []).append(sample)

        proposals: List[RegionalProposal] = []
        for group in groups.values():
            proposal = self._proposal_from_alarm_group(group)
            proposals.append(proposal)


            stop_group = [sample for sample in group if str(sample.event) == "STOP"]
            if stop_group and str(proposal.event) != "STOP":
                proposals.append(self._proposal_from_alarm_group(stop_group))

        proposals.sort(key=lambda item: item.quality, reverse=True)
        return proposals

    def _proposal_from_alarm_group(self, alarms: Sequence[AlarmSample]) -> RegionalProposal:
        peak = max(alarms, key=lambda item: item.score)
        member_track_ids = sorted({int(item.track_id) for item in alarms})
        alarm_packet_indices = sorted({int(item.packet_index) for item in alarms})


        frame_best: Dict[int, AlarmSample] = {}
        for sample in alarms:
            current = frame_best.get(sample.packet_index)
            if current is None or sample.score > current.score:
                frame_best[sample.packet_index] = sample
        frame_samples = list(frame_best.values())
        weights = np.asarray(
            [
                max(EPS, sample.confidence)
                * max(EPS, sample.score / max(peak.score, EPS))
                for sample in frame_samples
            ],
            dtype=float,
        )
        effective_count = _effective_sample_size(weights)
        recurrence = _continuous_sample_reliability(effective_count)
        mean_confidence = float(
            np.average(
                np.asarray([sample.confidence for sample in frame_samples], dtype=float),
                weights=weights,
            )
        )

        centers = np.asarray([sample.center_xy for sample in frame_samples], dtype=float)
        scales = np.asarray([max(1.0, sample.scale) for sample in frame_samples], dtype=float)
        center = np.average(centers, axis=0, weights=weights)
        normalized_distances = np.linalg.norm(centers - center[None, :], axis=1) / max(
            float(np.median(scales)), EPS
        )
        dispersion = float(np.average(normalized_distances, weights=weights))
        spatial_concentration = float(1.0 / (1.0 + dispersion))

        peak_excess = clip01((peak.score - self.threshold) / max(self.threshold, EPS))


        credibility = float(1.0 - (1.0 - recurrence) * (1.0 - peak_excess))
        support = max(recurrence, peak_excess)
        quality = float(
            peak.score
            * math.sqrt(
                max(
                    EPS,
                    support
                    * credibility
                    * mean_confidence
                    * spatial_concentration,
                )
            )
        )


        per_track: Dict[int, List[AlarmSample]] = {}
        for sample in alarms:
            per_track.setdefault(int(sample.track_id), []).append(sample)
        track_relevance: Dict[int, float] = {}
        for track_id, samples in per_track.items():
            track_peak = max(sample.score for sample in samples) / max(peak.score, EPS)
            track_weights = np.asarray(
                [
                    max(EPS, sample.confidence)
                    * max(EPS, sample.score / max(peak.score, EPS))
                    for sample in samples
                ],
                dtype=float,
            )
            track_recurrence = _continuous_sample_reliability(
                _effective_sample_size(track_weights)
            )
            track_centers = np.asarray([sample.center_xy for sample in samples], dtype=float)
            track_scales = np.asarray([max(1.0, sample.scale) for sample in samples], dtype=float)
            track_center = np.average(track_centers, axis=0, weights=track_weights)
            distance_to_peak = float(
                np.linalg.norm(track_center - np.asarray(peak.center_xy, dtype=float))
            )
            reference_scale = max(
                EPS,
                0.5 * (float(np.median(track_scales)) + max(1.0, peak.scale)),
            )
            spatial_affinity = 1.0 / (1.0 + distance_to_peak / reference_scale)
            track_relevance[track_id] = float(
                math.sqrt(
                    max(
                        EPS,
                        track_peak
                        * max(track_recurrence, track_peak)
                        * spatial_affinity,
                    )
                )
            )

        ranked_tracks = sorted(
            track_relevance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        relevance_values = np.asarray(
            [max(EPS, value) for _, value in ranked_tracks],
            dtype=float,
        )
        focus_count = min(
            len(ranked_tracks),
            max(
                1,
                int(
                    math.ceil(
                        math.sqrt(max(1.0, _effective_sample_size(relevance_values)))
                    )
                ),
            ),
        )
        focus_track_ids = [track_id for track_id, _ in ranked_tracks[:focus_count]]
        if int(peak.track_id) not in focus_track_ids:
            focus_track_ids = [int(peak.track_id)] + focus_track_ids
            focus_track_ids = focus_track_ids[: max(1, focus_count)]
        focus_track_ids = sorted(set(focus_track_ids))

        focus_alarms = [
            sample for sample in alarms if int(sample.track_id) in set(focus_track_ids)
        ]
        if not focus_alarms:
            focus_alarms = [peak]
        min_x = min(item.bbox_xyxy[0] for item in focus_alarms)
        min_y = min(item.bbox_xyxy[1] for item in focus_alarms)
        max_x = max(item.bbox_xyxy[2] for item in focus_alarms)
        max_y = max(item.bbox_xyxy[3] for item in focus_alarms)
        median_scale = float(np.median([item.scale for item in focus_alarms]))
        core_padding = median_scale
        core_roi = (
            min_x - core_padding,
            min_y - core_padding,
            max_x + core_padding,
            max_y + core_padding,
        )
        focus_centers = np.asarray(
            [sample.center_xy for sample in focus_alarms],
            dtype=float,
        )
        focus_weights = np.asarray(
            [
                max(EPS, sample.confidence)
                * max(EPS, sample.score / max(peak.score, EPS))
                for sample in focus_alarms
            ],
            dtype=float,
        )
        focus_center = np.average(focus_centers, axis=0, weights=focus_weights)
        return RegionalProposal(
            representative_track_id=int(peak.track_id),
            member_track_ids=member_track_ids,
            focus_track_ids=focus_track_ids,
            track_relevance=track_relevance,
            alarm_packet_indices=alarm_packet_indices,
            alarm_samples=list(alarms),
            event=str(peak.event),
            peak_score=float(peak.score),
            peak_packet_index=int(peak.packet_index),
            core_roi_xyxy=core_roi,
            core_center_xy=(float(focus_center[0]), float(focus_center[1])),
            core_scale=float(np.median([sample.scale for sample in focus_alarms])),
            effective_alarm_count=float(effective_count),
            recurrence_reliability=float(recurrence),
            spatial_concentration=spatial_concentration,
            mean_confidence=mean_confidence,
            credibility=credibility,
            quality=quality,
        )

    def _select_top_proposals(
        self,
        proposals: Sequence[RegionalProposal],
    ) -> List[RegionalProposal]:
        if not proposals:
            return []


        stop_pool = sorted(
            [item for item in proposals if str(item.event) == "STOP"],
            key=lambda item: item.quality,
            reverse=True,
        )
        persistent_pool = sorted(
            [item for item in proposals if _proposal_is_persistent_alarm(item)],
            key=lambda item: item.quality,
            reverse=True,
        )
        other_pool = sorted(
            [
                item
                for item in proposals
                if str(item.event) != "STOP" and not _proposal_is_persistent_alarm(item)
            ],
            key=lambda item: item.quality,
            reverse=True,
        )

        selected: List[RegionalProposal] = []
        seen: set[Tuple[str, int, int]] = set()

        def append_unique(items: Sequence[RegionalProposal]) -> None:
            for item in items:
                spatial_scale = max(20.0, float(item.core_scale))
                key = (
                    str(item.event),
                    int(round(item.core_center_xy[0] / spatial_scale)),
                    int(round(item.core_center_xy[1] / spatial_scale)),
                )
                if key in seen:
                    continue
                selected.append(item)
                seen.add(key)

        append_unique(stop_pool)
        if persistent_pool:
            persistent_qualities = np.asarray(
                [max(item.quality, EPS) for item in persistent_pool],
                dtype=float,
            )
            persistent_keep = min(
                len(persistent_pool),
                max(
                    1,
                    int(
                        math.ceil(
                            math.sqrt(
                                max(1.0, _effective_sample_size(persistent_qualities))
                            )
                        )
                    ),
                ),
            )
            append_unique(persistent_pool[:persistent_keep])

        if other_pool:
            qualities = np.asarray(
                [max(item.quality, EPS) for item in other_pool],
                dtype=float,
            )
            effective_event_count = _effective_sample_size(qualities)
            keep_count = min(
                len(other_pool),
                max(
                    1 if not selected else 0,
                    int(math.ceil(math.sqrt(max(1.0, effective_event_count)))),
                ),
            )
            append_unique(other_pool[:keep_count])

        selected.sort(key=lambda item: item.quality, reverse=True)
        return selected


    def _materialize_events(
        self,
        timeline: Sequence[CropFramePacket],
        proposal: RegionalProposal,
    ) -> List[AdaptiveEvent]:
        primary = self._materialize_event(timeline, proposal)
        if primary is None:
            return []

        if primary.stop_cause_search:
            alarm_event = self._materialize_stop_alarm_event(timeline, proposal, primary)
            return ([alarm_event] if alarm_event is not None else []) + [primary]

        events = [primary]


        if self._needs_forward_outcome(timeline, proposal, primary):
            forward_event = self._materialize_forward_outcome_event(
                timeline, proposal, primary
            )
            if forward_event is not None:
                events.append(forward_event)
        return events

    def _needs_forward_outcome(
        self,
        timeline: Sequence[CropFramePacket],
        proposal: RegionalProposal,
        event: AdaptiveEvent,
    ) -> bool:


        if str(proposal.event) not in {"IMPACT", "OVERSPEED", "LATERAL"}:
            return False
        multi_frame_support = bool(
            len(proposal.alarm_packet_indices) >= 3
            and proposal.effective_alarm_count >= 2.0
        )
        strong_short_support = bool(
            len(proposal.alarm_packet_indices) >= 2
            and proposal.peak_score >= 2.2 * self.threshold
        )
        if not (multi_frame_support or strong_short_support):
            return False
        peak_packet = timeline[proposal.peak_packet_index]
        peak_result = peak_packet.results.get(proposal.representative_track_id)
        if peak_result is None:
            return False
        lateral = max(0.0, float(peak_result.lateral_score))
        overspeed = max(0.0, float(peak_result.overspeed_score))
        decel = max(0.0, float(peak_result.decel_score))
        stop = max(0.0, float(peak_result.stop_score))
        if stop >= max(lateral, overspeed, decel):
            return False

        normal_lane = float(peak_result.debug.get("normal_lane_change_alternative", 0.0))
        side_slip = float(peak_result.debug.get("side_slip_likelihood", 0.0))
        noise = max(
            float(peak_result.debug.get("motion_resolution", 0.0)),
            float(peak_result.debug.get("sigma_parallel", 0.0)),
            EPS,
        )
        motion_signal = max(
            abs(float(peak_result.v_parallel)),
            abs(float(peak_result.v_perpendicular)),
            float(peak_result.debug.get("recent_speed", 0.0)),
        ) / noise


        if normal_lane >= 0.72 and side_slip < 0.35 and overspeed < 1.0 and decel < 0.6:
            return False
        return bool(motion_signal >= 1.0 or proposal.peak_score >= 1.6 * self.threshold)

    def _materialize_forward_outcome_event(
        self,
        timeline: Sequence[CropFramePacket],
        proposal: RegionalProposal,
        primary: AdaptiveEvent,
    ) -> Optional[AdaptiveEvent]:
        alarm_samples = sorted(proposal.alarm_samples, key=lambda item: item.packet_index)
        if not alarm_samples:
            return None
        anchor_sample = alarm_samples[-1]
        anchor_index = int(anchor_sample.packet_index)
        if not (0 <= anchor_index < len(timeline)):
            return None
        anchor_time = timeline[anchor_index].time_sec
        end_time = min(timeline[-1].time_sec, anchor_time + _FORWARD_OUTCOME_SECONDS)
        start_time = max(timeline[0].time_sec, anchor_time - 0.45)
        span = [
            index
            for index, packet in enumerate(timeline)
            if start_time - EPS <= packet.time_sec <= end_time + EPS
        ]
        if len(span) < 2 or timeline[span[-1]].time_sec <= anchor_time + 0.35:
            return None

        direction, speed_px_s, consistency = _estimate_forward_motion(
            timeline, proposal, anchor_index
        )
        if consistency < 0.20:
            return None
        keyframes, roles, phases = _select_forward_outcome_keyframes(
            timeline, span, anchor_index
        )
        if len(keyframes) < 2:
            return None
        roi, forward_center = _forward_outcome_roi(
            timeline=timeline,
            proposal=proposal,
            keyframe_indices=keyframes,
            anchor_index=anchor_index,
            direction=direction,
            speed_px_s=speed_px_s,
            horizon_seconds=max(0.0, timeline[span[-1]].time_sec - anchor_time),
        )
        focus_ids = list(dict.fromkeys([int(anchor_sample.track_id)] + list(proposal.focus_track_ids)))
        return AdaptiveEvent(
            representative_track_id=int(anchor_sample.track_id),
            member_track_ids=list(proposal.member_track_ids),
            focus_track_ids=focus_ids,
            event=str(proposal.event),
            peak_score=float(proposal.peak_score),
            peak_packet_index=anchor_index,
            span_start_index=span[0],
            span_end_index=span[-1],
            lookback_seconds=max(0.0, anchor_time - timeline[span[0]].time_sec),
            future_seconds=max(0.0, timeline[span[-1]].time_sec - anchor_time),
            keyframe_indices=keyframes,
            keyframe_roles=roles,
            phase_indices=phases,
            neighbor_ids=[],
            neighbor_weights={},
            roi_xyxy=roi,
            core_roi_xyxy=proposal.core_roi_xyxy,
            core_center_xy=forward_center,
            core_scale=proposal.core_scale,
            proposal_quality=proposal.quality,
            proposal_credibility=proposal.credibility,
            effective_alarm_count=proposal.effective_alarm_count,
            alarm_frame_count=len(proposal.alarm_packet_indices),
            stop_retrospective=False,
            stop_alarm_focus=False,
            stop_cause_search=False,
            persistent_alarm_focus=False,
            forward_outcome_search=True,
            temporal_mode="forward_outcome_search",
        )

    def _materialize_stop_alarm_event(
        self,
        timeline: Sequence[CropFramePacket],
        proposal: RegionalProposal,
        cause_event: AdaptiveEvent,
    ) -> Optional[AdaptiveEvent]:
        alarm_indices = sorted(
            int(index)
            for index in proposal.alarm_packet_indices
            if 0 <= int(index) < len(timeline)
        )
        if not alarm_indices:
            return None
        first_alarm = alarm_indices[0]
        last_alarm = alarm_indices[-1]
        peak_index = int(proposal.peak_packet_index)
        start_time = timeline[first_alarm].time_sec - 1.0
        end_time = max(timeline[last_alarm].time_sec, timeline[peak_index].time_sec) + 1.5
        span = [
            index for index, packet in enumerate(timeline)
            if start_time - EPS <= packet.time_sec <= end_time + EPS
        ]
        if not span:
            return None
        keyframes, roles, phases = _select_stop_alarm_keyframes(
            timeline, span, alarm_indices, peak_index
        )
        roi = _stop_alarm_context_roi(timeline, proposal, keyframes)
        return AdaptiveEvent(
            representative_track_id=proposal.representative_track_id,
            member_track_ids=list(proposal.member_track_ids),
            focus_track_ids=list(proposal.focus_track_ids),
            event="STOP",
            peak_score=proposal.peak_score,
            peak_packet_index=peak_index,
            span_start_index=span[0],
            span_end_index=span[-1],
            lookback_seconds=max(0.0, timeline[peak_index].time_sec - timeline[span[0]].time_sec),
            future_seconds=max(0.0, timeline[span[-1]].time_sec - timeline[peak_index].time_sec),
            keyframe_indices=keyframes,
            keyframe_roles=roles,
            phase_indices=phases,
            neighbor_ids=[],
            neighbor_weights={},
            roi_xyxy=roi,
            core_roi_xyxy=proposal.core_roi_xyxy,
            core_center_xy=proposal.core_center_xy,
            core_scale=proposal.core_scale,
            proposal_quality=proposal.quality,
            proposal_credibility=proposal.credibility,
            effective_alarm_count=proposal.effective_alarm_count,
            alarm_frame_count=len(proposal.alarm_packet_indices),
            stop_retrospective=False,
            stop_alarm_focus=True,
            stop_cause_search=False,
            persistent_alarm_focus=False,
            forward_outcome_search=False,
            temporal_mode="stop_alarm_focus",
        )


    def _materialize_event(
        self,
        timeline: Sequence[CropFramePacket],
        proposal: RegionalProposal,
    ) -> Optional[AdaptiveEvent]:
        if not timeline:
            return None
        peak_packet = timeline[proposal.peak_packet_index]
        peak_result = peak_packet.results.get(proposal.representative_track_id)
        if peak_result is None:
            return None

        lateral = max(0.0, float(peak_result.lateral_score))
        overspeed = max(0.0, float(peak_result.overspeed_score))
        decel = max(0.0, float(peak_result.decel_score))
        stop = max(0.0, float(peak_result.stop_score))
        branch_total = lateral + overspeed + decel + stop + EPS
        delayed_ratio = clip01((decel + stop) / branch_total)
        forward_ratio = clip01(overspeed / branch_total)
        recurrence = clip01(proposal.recurrence_reliability)


        stop_retrospective = bool(
            str(peak_result.event) == "STOP"
            or int(peak_result.debug.get("stop_semantic_dominant", 0)) == 1
            or (stop >= max(lateral, overspeed, decel))
        )


        lookback_seconds = self.focus_base_seconds * (
            1.0 + math.sqrt(max(recurrence, delayed_ratio))
        )
        future_seconds = self.focus_base_seconds * math.sqrt(
            max(forward_ratio, recurrence)
        )

        persistent_alarm_focus = bool(
            (not stop_retrospective) and _proposal_is_persistent_alarm(proposal)
        )


        if stop_retrospective and timeline:
            available_history = max(0.0, peak_packet.time_sec - timeline[0].time_sec)
            lookback_seconds = max(
                lookback_seconds,
                min(self.stop_lookback_seconds, available_history),
            )
            future_seconds = max(future_seconds, min(self.focus_base_seconds, 2.0))
        elif persistent_alarm_focus:


            alarm_times = [
                timeline[index].time_sec
                for index in proposal.alarm_packet_indices
                if 0 <= int(index) < len(timeline)
            ]
            alarm_duration = (
                max(alarm_times) - min(alarm_times)
                if len(alarm_times) >= 2
                else 0.0
            )
            lookback_seconds = max(
                lookback_seconds,
                self.focus_base_seconds
                + alarm_duration
                + self.focus_base_seconds * math.sqrt(max(recurrence, 0.25)),
            )
            future_seconds = max(
                future_seconds,
                min(
                    2.0 * self.focus_base_seconds,
                    self.focus_base_seconds + 0.75 * alarm_duration,
                ),
            )

        start_time = peak_packet.time_sec - lookback_seconds
        end_time = peak_packet.time_sec + future_seconds
        candidate_indices = [
            index
            for index, packet in enumerate(timeline)
            if start_time - EPS <= packet.time_sec <= end_time + EPS
        ]
        if not candidate_indices:
            return None


        span_start = candidate_indices[0]
        span_end = candidate_indices[-1]
        span_indices = list(range(span_start, span_end + 1))


        if stop_retrospective:
            keyframe_indices, keyframe_roles, phase_indices = (
                _select_stop_cause_keyframes(
                    timeline,
                    span_indices,
                    proposal,
                    proposal.peak_packet_index,
                )
            )
            temporal_mode = "stop_cause_search"
        elif persistent_alarm_focus:
            keyframe_indices, keyframe_roles, phase_indices = (
                _select_persistent_alarm_keyframes(
                    timeline,
                    span_indices,
                    proposal,
                    proposal.peak_packet_index,
                )
            )
            temporal_mode = "persistent_alarm_focus"
        else:
            keyframe_indices, keyframe_roles, phase_indices = _select_regional_keyframes(
                timeline,
                proposal,
                span_indices,
                proposal.core_roi_xyxy,
            )
            temporal_mode = "phase_focus"
        if not keyframe_indices:
            return None

        if stop_retrospective:
            neighbor_ids, neighbor_weights = [], {}
            roi = _stop_cause_context_roi(
                timeline,
                proposal,
                keyframe_indices,
            )
        else:
            neighbor_ids, neighbor_weights = _select_relevant_neighbors_for_region(
                timeline,
                proposal,
                span_indices,
            )
            roi = _adaptive_regional_roi(
                timeline,
                proposal,
                span_indices,
                keyframe_indices,
                neighbor_ids,
                lookback_seconds,
                future_seconds,
            )

        return AdaptiveEvent(
            representative_track_id=proposal.representative_track_id,
            member_track_ids=list(proposal.member_track_ids),
            focus_track_ids=list(proposal.focus_track_ids),
            event=proposal.event,
            peak_score=proposal.peak_score,
            peak_packet_index=proposal.peak_packet_index,
            span_start_index=span_start,
            span_end_index=span_end,
            lookback_seconds=float(lookback_seconds),
            future_seconds=float(future_seconds),
            keyframe_indices=keyframe_indices,
            keyframe_roles=keyframe_roles,
            phase_indices=phase_indices,
            neighbor_ids=neighbor_ids,
            neighbor_weights=neighbor_weights,
            roi_xyxy=roi,
            core_roi_xyxy=proposal.core_roi_xyxy,
            core_center_xy=proposal.core_center_xy,
            core_scale=proposal.core_scale,
            proposal_quality=proposal.quality,
            proposal_credibility=proposal.credibility,
            effective_alarm_count=proposal.effective_alarm_count,
            alarm_frame_count=len(proposal.alarm_packet_indices),
            stop_retrospective=bool(stop_retrospective),
            stop_alarm_focus=False,
            stop_cause_search=bool(stop_retrospective),
            persistent_alarm_focus=bool(persistent_alarm_focus),
            forward_outcome_search=False,
            temporal_mode=str(temporal_mode),
        )

    def _is_recent_duplicate(
        self,
        timeline: Sequence[CropFramePacket],
        event: AdaptiveEvent,
    ) -> bool:
        peak_time = timeline[event.peak_packet_index].time_sec
        for previous in self.recent_emissions:


            continuation_slack = self.focus_base_seconds * math.sqrt(
                clip01(event.proposal_credibility)
            )
            center_gap = math.hypot(
                event.core_center_xy[0] - previous.core_center_xy[0],
                event.core_center_xy[1] - previous.core_center_xy[1],
            )
            same_region = center_gap <= max(
                20.0,
                0.5 * (event.core_scale + previous.core_scale),
            )
            if (
                event.temporal_mode == previous.temporal_mode
                and peak_time <= previous.end_time_sec + continuation_slack + EPS
                and same_region
            ):
                return True
        return False

    def _remember_emission(
        self,
        timeline: Sequence[CropFramePacket],
        event: AdaptiveEvent,
    ) -> None:
        self.recent_emissions.append(
            RecentEmission(
                roi_xyxy=event.roi_xyxy,
                core_roi_xyxy=event.core_roi_xyxy,
                core_center_xy=event.core_center_xy,
                core_scale=event.core_scale,
                start_time_sec=timeline[event.span_start_index].time_sec,
                end_time_sec=timeline[event.span_end_index].time_sec,
                peak_time_sec=timeline[event.peak_packet_index].time_sec,
                quality=event.proposal_quality,
                temporal_mode=event.temporal_mode,
            )
        )


    def _write_event(
        self,
        timeline: Sequence[CropFramePacket],
        event: AdaptiveEvent,
        segment_dir: Path,
        segment_index: int,
        segment_start: float,
        segment_end: float,
        local_index: int,
    ) -> Tuple[Dict[str, Path], np.ndarray, EventCropArtifact]:
        self.event_serial += 1
        peak_packet = timeline[event.peak_packet_index]
        stem = (
            f"event_{local_index:02d}_region_track_{event.representative_track_id:05d}_"
            f"{event.event}_{event.temporal_mode}_peak_{peak_packet.time_sec:08.3f}s"
        ).replace(".", "p")
        clean_path = segment_dir / f"{stem}_clean.jpg"
        debug_path = segment_dir / f"{stem}_debug.jpg"
        vlm_storyboard_path = segment_dir / f"{stem}_vlm_storyboard.jpg"
        vlm_comparison_path = segment_dir / f"{stem}_vlm_comparison.jpg"
        context_clean_path = segment_dir / f"{stem}_context_clean.jpg"
        context_debug_path = segment_dir / f"{stem}_context_debug.jpg"
        timeline_clean_path = segment_dir / f"{stem}_timeline_clean.jpg"
        timeline_debug_path = segment_dir / f"{stem}_timeline_debug.jpg"
        json_path = segment_dir / f"{stem}.json"
        frames_dir = segment_dir / f"{stem}_frames"
        ensure_dir(frames_dir)

        focus_clean_tiles: List[np.ndarray] = []
        vlm_focus_tiles: List[np.ndarray] = []
        focus_debug_tiles: List[np.ndarray] = []
        context_clean_tiles: List[np.ndarray] = []
        context_debug_tiles: List[np.ndarray] = []
        clean_frame_paths: List[Path] = []
        debug_frame_paths: List[Path] = []
        keyframe_metadata: List[Dict[str, Any]] = []

        context_x1, context_y1, context_x2, context_y2 = event.roi_xyxy
        focus_set = set(event.focus_track_ids)
        fixed_focus_roi = None
        if event.stop_alarm_focus:
            fixed_focus_roi = _fixed_stop_alarm_focus_roi(timeline, event)
        elif event.stop_cause_search:
            fixed_focus_roi = _fixed_stop_cause_focus_roi(timeline, event)
        elif event.persistent_alarm_focus:
            fixed_focus_roi = _fixed_persistent_alarm_focus_roi(timeline, event)
        elif event.forward_outcome_search:
            fixed_focus_roi = event.roi_xyxy

        for order, packet_index in enumerate(event.keyframe_indices, start=1):
            packet = timeline[packet_index]
            frame = packet.decode()
            focus_roi = (
                fixed_focus_roi
                if fixed_focus_roi is not None
                else _frame_focus_roi(packet, event)
            )
            fx1, fy1, fx2, fy2 = focus_roi

            focus_clean = frame[fy1:fy2, fx1:fx2].copy()
            vlm_focus = _prepare_vlm_focus_image(focus_clean)
            focus_debug = focus_clean.copy()
            context_clean = frame[
                context_y1:context_y2,
                context_x1:context_x2,
            ].copy()
            context_debug = context_clean.copy()


            for member_id in event.member_track_ids:
                target = packet.observations.get(member_id)
                if target is None:
                    continue
                if member_id in focus_set:
                    _draw_labeled_box_in_crop(
                        focus_debug,
                        target["bbox_xyxy"],
                        focus_roi,
                        (0, 0, 255),
                        3 if member_id == event.representative_track_id else 2,
                        f"focus id={member_id}",
                    )
                _draw_labeled_box_in_crop(
                    context_debug,
                    target["bbox_xyxy"],
                    event.roi_xyxy,
                    (0, 0, 255) if member_id in focus_set else (0, 120, 255),
                    3 if member_id == event.representative_track_id else 2,
                    (
                        f"focus id={member_id}"
                        if member_id in focus_set
                        else f"region id={member_id}"
                    ),
                )

            for neighbor_id in event.neighbor_ids:
                if neighbor_id in event.member_track_ids:
                    continue
                neighbor = packet.observations.get(neighbor_id)
                if neighbor is None:
                    continue
                if _bbox_matches_region(neighbor["bbox_xyxy"], focus_roi):
                    _draw_labeled_box_in_crop(
                        focus_debug,
                        neighbor["bbox_xyxy"],
                        focus_roi,
                        (0, 170, 255),
                        2,
                        f"ctx {neighbor_id}",
                    )
                _draw_labeled_box_in_crop(
                    context_debug,
                    neighbor["bbox_xyxy"],
                    event.roi_xyxy,
                    (0, 170, 255),
                    2,
                    f"ctx {neighbor_id}",
                )

            result_track_id, result = _best_region_result(
                packet,
                event.focus_track_ids,
                focus_roi,
            )
            if result is None:
                result_track_id, result = _best_region_result(
                    packet,
                    event.member_track_ids,
                    event.roi_xyxy,
                )
            axis_observation = (
                packet.observations.get(result_track_id)
                if result_track_id is not None
                else None
            )
            if (
                result is not None
                and axis_observation is not None
                and result.road_axis.ready
            ):
                _draw_axis_in_crop(
                    focus_debug,
                    axis_observation["bbox_xyxy"],
                    focus_roi,
                    result,
                )
                _draw_axis_in_crop(
                    context_debug,
                    axis_observation["bbox_xyxy"],
                    event.roi_xyxy,
                    result,
                )

            score = float(result.score) if result is not None else 0.0
            role = event.keyframe_roles.get(
                int(packet_index),
                "peak"
                if int(packet_index) == int(event.peak_packet_index)
                else "transition",
            )
            label = (
                f"{order}/{len(event.keyframe_indices)}  "
                f"t={packet.time_sec:.2f}s  f={packet.frame_id}  "
                f"role={role}  "
                f"{result.event if result else event.event}  S={score:.2f}"
            )
            _outlined_text(focus_debug, label, (8, 22), 0.44)
            _outlined_text(context_debug, label, (8, 22), 0.44)

            frame_stem = (
                f"frame_{order:02d}_{role}_t_{packet.time_sec:08.3f}s_f_{packet.frame_id:08d}"
            ).replace(".", "p")
            focus_clean_path = frames_dir / f"{frame_stem}_focus.jpg"
            vlm_focus_path = frames_dir / f"{frame_stem}_vlm.jpg"
            focus_debug_path = frames_dir / f"{frame_stem}_focus_debug.jpg"
            context_frame_path = frames_dir / f"{frame_stem}_context.jpg"
            context_frame_debug_path = frames_dir / f"{frame_stem}_context_debug.jpg"

            cv2.imwrite(
                str(focus_clean_path),
                focus_clean,
                [int(cv2.IMWRITE_JPEG_QUALITY), 94],
            )
            cv2.imwrite(
                str(vlm_focus_path),
                vlm_focus,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            cv2.imwrite(
                str(focus_debug_path),
                focus_debug,
                [int(cv2.IMWRITE_JPEG_QUALITY), 94],
            )
            cv2.imwrite(
                str(context_frame_path),
                context_clean,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            cv2.imwrite(
                str(context_frame_debug_path),
                context_debug,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            clean_frame_paths.append(vlm_focus_path.resolve())
            debug_frame_paths.append(focus_debug_path.resolve())

            focus_clean_tiles.append(focus_clean)
            vlm_focus_tiles.append(vlm_focus)
            focus_debug_tiles.append(focus_debug)
            context_clean_tiles.append(context_clean)
            context_debug_tiles.append(context_debug)

            keyframe_metadata.append(
                {
                    "order": order,
                    "role": role,
                    "is_peak": int(packet_index) == int(event.peak_packet_index),
                    "recommended_for_vlm": True,

                    "vlm_image": str(vlm_focus_path.relative_to(segment_dir)),
                    "focus_image": str(focus_clean_path.relative_to(segment_dir)),
                    "clean_image": str(vlm_focus_path.relative_to(segment_dir)),
                    "debug_image": str(focus_debug_path.relative_to(segment_dir)),
                    "context_image": str(context_frame_path.relative_to(segment_dir)),
                    "context_debug_image": str(
                        context_frame_debug_path.relative_to(segment_dir)
                    ),
                    "focus_roi_xyxy": list(focus_roi),
                    "context_roi_xyxy": list(event.roi_xyxy),
                    "frame_id": packet.frame_id,
                    "time_sec": packet.time_sec,
                    "regional_score": score,
                    "event": str(result.event) if result is not None else event.event,
                    "representative_track_id": result_track_id,
                    "focus_track_ids": event.focus_track_ids,
                    "member_bboxes": {
                        str(track_id): list(packet.observations[track_id]["bbox_xyxy"])
                        for track_id in event.member_track_ids
                        if track_id in packet.observations
                    },
                    "branch_scores": (
                        {
                            "lateral": float(result.lateral_score),
                            "overspeed": float(result.overspeed_score),
                            "decel": float(result.decel_score),
                            "stop": float(result.stop_score),
                            "impact": float(result.debug.get("impact_episode_score", 0.0)),
                        }
                        if result is not None
                        else None
                    ),
                }
            )

        title = (
            f"focus={','.join(str(v) for v in event.focus_track_ids)}  "
            f"region_tracks={len(event.member_track_ids)}  "
            f"event={event.event}  peak={event.peak_score:.2f}  "
            f"mode={event.temporal_mode}  "
            f"adaptive={timeline[event.span_start_index].time_sec:.2f}s-"
            f"{timeline[event.span_end_index].time_sec:.2f}s"
        )
        focus_clean_canvas = _compose_image_grid(
            vlm_focus_tiles,
            max_side=self.canvas_max_side,
            title=None,
            background=(0, 0, 0),
        )
        focus_debug_canvas = _compose_image_grid(
            focus_debug_tiles,
            max_side=self.canvas_max_side,
            title=title,
            background=(24, 24, 24),
        )
        context_clean_canvas = _compose_image_grid(
            context_clean_tiles,
            max_side=self.canvas_max_side,
            title=None,
            background=(0, 0, 0),
        )
        context_debug_canvas = _compose_image_grid(
            context_debug_tiles,
            max_side=self.canvas_max_side,
            title=f"context  {title}",
            background=(24, 24, 24),
        )
        cv2.imwrite(str(clean_path), focus_clean_canvas)
        cv2.imwrite(str(debug_path), focus_debug_canvas)
        cv2.imwrite(str(context_clean_path), context_clean_canvas)
        cv2.imwrite(str(context_debug_path), context_debug_canvas)


        vlm_storyboard = _compose_vlm_storyboard(
            vlm_focus_tiles,
            max_side=self.canvas_max_side,
        )
        cv2.imwrite(
            str(vlm_storyboard_path),
            vlm_storyboard,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        comparison_positions = _select_vlm_comparison_positions(
            keyframe_metadata,
            event.temporal_mode,
            vlm_focus_tiles,
        )
        comparison_tiles = [vlm_focus_tiles[index] for index in comparison_positions]
        if comparison_tiles:
            vlm_comparison = _compose_vlm_storyboard(
                comparison_tiles,
                max_side=min(self.canvas_max_side, 960),
                force_single_row=True,
            )
            cv2.imwrite(
                str(vlm_comparison_path),
                vlm_comparison,
                [int(cv2.IMWRITE_JPEG_QUALITY), 96],
            )

        audit_indices: List[int] = []
        if event.stop_cause_search and fixed_focus_roi is not None:
            audit_indices = _sample_stop_audit_indices(
                timeline,
                list(range(event.span_start_index, event.span_end_index + 1)),
            )
            audit_clean_tiles: List[np.ndarray] = []
            audit_debug_tiles: List[np.ndarray] = []
            for audit_order, audit_index in enumerate(audit_indices, start=1):
                audit_packet = timeline[audit_index]
                audit_frame = audit_packet.decode()
                ax1, ay1, ax2, ay2 = fixed_focus_roi
                audit_clean = audit_frame[ay1:ay2, ax1:ax2].copy()
                audit_debug = audit_clean.copy()
                for member_id in event.member_track_ids:
                    observation = audit_packet.observations.get(member_id)
                    if observation is None:
                        continue
                    _draw_labeled_box_in_crop(
                        audit_debug,
                        observation["bbox_xyxy"],
                        fixed_focus_roi,
                        (0, 0, 255),
                        2,
                        f"id={member_id}",
                    )
                _outlined_text(
                    audit_debug,
                    f"audit {audit_order}/{len(audit_indices)}  t={audit_packet.time_sec:.2f}s",
                    (8, 22),
                    0.44,
                )
                audit_clean_tiles.append(audit_clean)
                audit_debug_tiles.append(audit_debug)

            timeline_clean_canvas = _compose_image_grid(
                audit_clean_tiles,
                max_side=self.canvas_max_side,
                title=None,
                background=(0, 0, 0),
            )
            timeline_debug_canvas = _compose_image_grid(
                audit_debug_tiles,
                max_side=self.canvas_max_side,
                title=(
                    f"STOP audit timeline  {timeline[event.span_start_index].time_sec:.2f}s-"
                    f"{timeline[event.span_end_index].time_sec:.2f}s  "
                    f"VLM uses only {len(event.keyframe_indices)} causal frames"
                ),
                background=(24, 24, 24),
            )
            cv2.imwrite(str(timeline_clean_path), timeline_clean_canvas)
            cv2.imwrite(str(timeline_debug_path), timeline_debug_canvas)

        peak_result = peak_packet.results[event.representative_track_id]
        metadata = {
            "segment_index": segment_index,
            "segment_start_sec": segment_start,
            "segment_end_sec": segment_end,
            "event_index": self.event_serial,
            "representative_track_id": event.representative_track_id,
            "member_track_ids": event.member_track_ids,
            "focus_track_ids": event.focus_track_ids,
            "event": event.event,
            "peak_score": event.peak_score,
            "peak_frame_id": peak_packet.frame_id,
            "peak_time_sec": peak_packet.time_sec,
            "proposal_quality": event.proposal_quality,
            "proposal_credibility": event.proposal_credibility,
            "effective_alarm_count": event.effective_alarm_count,
            "alarm_frame_count": event.alarm_frame_count,
            "temporal_mode": event.temporal_mode,
            "stop_retrospective": event.stop_retrospective,
            "stop_alarm_focus": event.stop_alarm_focus,
            "stop_cause_search": event.stop_cause_search,
            "persistent_alarm_focus": event.persistent_alarm_focus,
            "forward_outcome_search": event.forward_outcome_search,
            "adaptive_start_frame_id": timeline[event.span_start_index].frame_id,
            "adaptive_start_sec": timeline[event.span_start_index].time_sec,
            "adaptive_end_frame_id": timeline[event.span_end_index].frame_id,
            "adaptive_end_sec": timeline[event.span_end_index].time_sec,
            "lookback_seconds": event.lookback_seconds,
            "future_seconds": event.future_seconds,
            "roi_xyxy": list(event.roi_xyxy),
            "fixed_focus_roi_xyxy": (
                list(fixed_focus_roi)
                if fixed_focus_roi is not None
                else None
            ),

            "vlm_sequence_image": str(vlm_storyboard_path.relative_to(segment_dir)),
            "vlm_storyboard_image": str(vlm_storyboard_path.relative_to(segment_dir)),
            "vlm_storyboard_frame_count": int(len(vlm_focus_tiles)),
            "vlm_comparison_image": (
                str(vlm_comparison_path.relative_to(segment_dir))
                if vlm_comparison_path.is_file()
                else ""
            ),
            "vlm_comparison_frame_count": int(len(comparison_positions)),
            "vlm_comparison_orders": [int(index + 1) for index in comparison_positions],
            "vlm_frame_order": "left_to_right_then_top_to_bottom",
            "vlm_tiles_are_separate_times": True,
            "context_canvas": str(context_clean_path.relative_to(segment_dir)),
            "context_debug_canvas": str(context_debug_path.relative_to(segment_dir)),
            "audit_timeline_canvas": (
                str(timeline_clean_path.relative_to(segment_dir))
                if audit_indices
                else ""
            ),
            "audit_timeline_debug_canvas": (
                str(timeline_debug_path.relative_to(segment_dir))
                if audit_indices
                else ""
            ),
            "audit_timeline_times_sec": [
                float(timeline[index].time_sec) for index in audit_indices
            ],
            "neighbor_ids": event.neighbor_ids,
            "neighbor_weights": {str(k): v for k, v in event.neighbor_weights.items()},
            "road_axis": {
                "x": peak_result.road_axis.x,
                "y": peak_result.road_axis.y,
                "confidence": peak_result.road_axis.confidence,
                "source": peak_result.road_axis.source,
            },
            "branch_scores_at_peak": {
                "lateral": peak_result.lateral_score,
                "overspeed": peak_result.overspeed_score,
                "decel": peak_result.decel_score,
                "stop": peak_result.stop_score,
                "impact": float(peak_result.debug.get("impact_episode_score", 0.0)),
            },
            "temporal_strategy": {
                "description": (
                    "STOP 当前确认事件：只覆盖首次报警、停车峰值和短时后果，用紧凑固定 ROI 确认车辆是否持续静止。"
                    if event.stop_alarm_focus
                    else
                    "STOP 原因搜索事件：在首次停车报警前约 10~12 秒内较密集采帧，最后加入首次报警和停车峰值，寻找碰撞、横置车体、烟雾或突然车流压缩。"
                    if event.stop_cause_search
                    else
                    "近期持续局部报警事件覆盖异常前、异常开始、持续发展、峰值以及后续状态。"
                    if event.persistent_alarm_focus
                    else
                    "道路前方结果搜索事件从最后一次局部报警开始，沿真实轨迹优先的运动方向继续观察约 2.4 秒，寻找超车/快速横移后才出现的碰撞或失控后果。"
                    if event.forward_outcome_search
                    else
                    "普通事件按触发前、前兆、交互前后、后果和结束阶段选择关键帧。"
                ),
                "phase_packet_indices": {
                    key: int(value) for key, value in event.phase_indices.items()
                },
                "selected_keyframes": keyframe_metadata,
            },
            "spatial_strategy": {
                "description": (
                    "VLM 只使用红框热点的固定紧凑 focus 图和其轻度增强版本；context 图只供人工审计。"
                    "STOP 当前确认不使用黄框邻车；STOP 原因搜索仅沿道路前后稍微扩大，仍不由黄框位置决定。"
                    "持续局部报警使用固定 ROI；道路前方结果搜索使用从报警热点到预测前方位置的固定紧凑走廊，不加入黄框邻车；普通事件最多加入一辆真正贴近的交互车辆。"
                )
            },
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifact = EventCropArtifact(
            segment_index=int(segment_index),
            segment_start_sec=float(segment_start),
            segment_end_sec=float(segment_end),
            event_index=int(self.event_serial),
            representative_track_id=int(event.representative_track_id),
            member_track_ids=tuple(int(v) for v in event.member_track_ids),
            event=str(event.event),
            peak_score=float(event.peak_score),
            peak_time_sec=float(peak_packet.time_sec),
            adaptive_start_sec=float(timeline[event.span_start_index].time_sec),
            adaptive_end_sec=float(timeline[event.span_end_index].time_sec),
            metadata_path=json_path.resolve(),
            clean_canvas_path=clean_path.resolve(),
            debug_canvas_path=debug_path.resolve(),
            keyframe_clean_paths=tuple(clean_frame_paths),
            keyframe_debug_paths=tuple(debug_frame_paths),
        )
        return {
            "clean": clean_path,
            "debug": debug_path,
            "context_clean": context_clean_path,
            "context_debug": context_debug_path,
            "json": json_path,
        }, focus_debug_canvas, artifact


def _event_vlm_priority(event: AdaptiveEvent) -> float:


    mode_bonus = {
        "stop_alarm_focus": 5.0,
        "stop_cause_search": 4.7,
        "forward_outcome_search": 4.6,
        "persistent_alarm_focus": 3.6,
        "phase_focus": 1.8,
    }.get(str(event.temporal_mode), 1.5)
    recurrence = _continuous_sample_reliability(event.effective_alarm_count)
    multi_frame = min(1.0, max(0.0, (event.alarm_frame_count - 1) / 3.0))
    return float(
        mode_bonus
        + math.log1p(max(0.0, event.peak_score))
        + 0.75 * clip01(event.proposal_credibility)
        + 0.45 * recurrence
        + 0.30 * multi_frame
    )

def _alarm_samples_share_region(a: AlarmSample, b: AlarmSample) -> bool:
    dx = float(a.center_xy[0] - b.center_xy[0])
    dy = float(a.center_xy[1] - b.center_xy[1])
    pair_scale = max(20.0, 0.5 * (a.scale + b.scale))
    return bool(math.hypot(dx, dy) <= pair_scale)


def _continuous_sample_reliability(effective_count: float) -> float:
    count = max(0.0, float(effective_count))
    if count <= 0.0:
        return 0.0
    return float(1.0 - 1.0 / math.sqrt(count + 1.0))


def _proposal_is_persistent_alarm(proposal: RegionalProposal) -> bool:


    if str(proposal.event) == "STOP":
        return False
    return bool(
        len(proposal.alarm_packet_indices) >= 3
        and proposal.effective_alarm_count >= 2.0
    )


def _effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 0.0
    return float(np.sum(values) ** 2 / max(np.sum(values * values), EPS))


def _regions_touch(
    first: Sequence[float],
    second: Sequence[float],
) -> bool:
    ax1, ay1, ax2, ay2 = [float(v) for v in first]
    bx1, by1, bx2, by2 = [float(v) for v in second]
    gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0.0, max(ay1, by1) - min(ay2, by2))
    scale_a = math.sqrt(max(1.0, (ax2 - ax1) * (ay2 - ay1)))
    scale_b = math.sqrt(max(1.0, (bx2 - bx1) * (by2 - by1)))
    return bool(math.hypot(gap_x, gap_y) <= max(20.0, 0.5 * (scale_a + scale_b)))


def _bbox_center_scale(bbox: Sequence[float]) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        math.sqrt(max(1.0, (x2 - x1) * (y2 - y1))),
    )


def _bbox_matches_region(
    bbox: Sequence[float],
    roi: Sequence[float],
) -> bool:
    cx, cy, scale = _bbox_center_scale(bbox)
    x1, y1, x2, y2 = [float(v) for v in roi]
    nearest_x = min(max(cx, x1), x2)
    nearest_y = min(max(cy, y1), y2)
    return bool(math.hypot(cx - nearest_x, cy - nearest_y) <= max(20.0, scale))


def _best_region_result(
    packet: CropFramePacket,
    member_track_ids: Sequence[int],
    roi_xyxy: Sequence[float],
) -> Tuple[Optional[int], Optional[AnomalyResult]]:
    candidates: List[Tuple[float, int, AnomalyResult]] = []
    for track_id in member_track_ids:
        result = packet.results.get(track_id)
        if result is not None and track_id in packet.observations:
            candidates.append((float(result.score), int(track_id), result))
    if not candidates:
        for track_id, observation in packet.observations.items():
            if not _bbox_matches_region(observation["bbox_xyxy"], roi_xyxy):
                continue
            result = packet.results.get(track_id)
            if result is not None:
                candidates.append((float(result.score), int(track_id), result))
    if not candidates:
        return None, None
    _, track_id, result = max(candidates, key=lambda item: item[0])
    return track_id, result


def _bbox_contact_affinity(
    first_bbox: Sequence[float],
    second_bbox: Sequence[float],
) -> float:


    ax1, ay1, ax2, ay2 = [float(v) for v in first_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in second_bbox]
    gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0.0, max(ay1, by1) - min(ay2, by2))
    scale_a = math.sqrt(max(1.0, (ax2 - ax1) * (ay2 - ay1)))
    scale_b = math.sqrt(max(1.0, (bx2 - bx1) * (by2 - by1)))
    normalized_gap = math.hypot(gap_x, gap_y) / max(EPS, 0.5 * (scale_a + scale_b))
    return float(1.0 / (1.0 + normalized_gap))


def _branch_vector(result: Optional[AnomalyResult]) -> Tuple[float, float, float, float, float]:
    if result is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        max(0.0, float(result.lateral_score)),
        max(0.0, float(result.overspeed_score)),
        max(0.0, float(result.decel_score)),
        max(0.0, float(result.stop_score)),
        max(0.0, float(result.debug.get("impact_episode_score", result.score if result.event == "IMPACT" else 0.0))),
    )


def _regional_state(
    packet: CropFramePacket,
    proposal: RegionalProposal,
    roi_xyxy: Sequence[float],
) -> Dict[str, Any]:


    candidate_ids = [
        track_id
        for track_id in proposal.focus_track_ids
        if track_id in packet.observations
    ]
    if not candidate_ids:
        candidate_ids = [
            track_id
            for track_id in proposal.member_track_ids
            if track_id in packet.observations
        ]
    if not candidate_ids:
        candidate_ids = [
            int(track_id)
            for track_id, observation in packet.observations.items()
            if _bbox_matches_region(observation["bbox_xyxy"], proposal.core_roi_xyxy)
        ]

    rows: List[Tuple[int, Dict[str, Any], Optional[AnomalyResult], float]] = []
    for track_id in candidate_ids:
        observation = packet.observations.get(track_id)
        if observation is None:
            continue
        result = packet.results.get(track_id)
        relevance = float(proposal.track_relevance.get(track_id, 0.0))
        score = float(result.score) if result is not None else 0.0
        weight = max(
            EPS,
            float(observation.get("det_conf", 0.0))
            * max(relevance, score / max(proposal.peak_score, EPS), EPS),
        )
        rows.append((track_id, observation, result, weight))

    if not rows:
        x1, y1, x2, y2 = [float(v) for v in roi_xyxy]
        return {
            "center_x": 0.5 * (x1 + x2),
            "center_y": 0.5 * (y1 + y2),
            "scale": math.sqrt(max(1.0, (x2 - x1) * (y2 - y1))),
            "score": 0.0,
            "branches": (0.0, 0.0, 0.0, 0.0, 0.0),
            "contact_affinity": 0.0,
            "track_id": 0,
        }

    weights = np.asarray([row[3] for row in rows], dtype=float)
    centers = []
    scales = []
    for _, observation, _, _ in rows:
        cx, cy, scale = _bbox_center_scale(observation["bbox_xyxy"])
        centers.append((cx, cy))
        scales.append(scale)
    center = np.average(np.asarray(centers, dtype=float), axis=0, weights=weights)
    scale = float(np.exp(np.average(np.log(np.maximum(np.asarray(scales), 1.0)), weights=weights)))

    best_track_id, best_observation, best_result, _ = max(
        rows,
        key=lambda row: (
            float(row[2].score) if row[2] is not None else 0.0,
            row[3],
        ),
    )
    contact_affinity = 0.0
    focus_set = {row[0] for row in rows}
    for track_id, observation, _, _ in rows:
        neighbor_ids = packet.neighbor_map.get(track_id, [])
        for neighbor_id in neighbor_ids:
            if neighbor_id in focus_set:
                continue
            neighbor = packet.observations.get(neighbor_id)
            if neighbor is None:
                continue
            contact_affinity = max(
                contact_affinity,
                _bbox_contact_affinity(
                    observation["bbox_xyxy"],
                    neighbor["bbox_xyxy"],
                ),
            )

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "scale": scale,
        "score": float(best_result.score) if best_result is not None else 0.0,
        "branches": _branch_vector(best_result),
        "contact_affinity": float(contact_affinity),
        "track_id": int(best_track_id),
        "bbox_xyxy": tuple(float(v) for v in best_observation["bbox_xyxy"]),
    }


def _stop_hotspot_packet_state(
    packet: CropFramePacket,
    proposal: RegionalProposal,
) -> Dict[str, float]:


    core_x, core_y = proposal.core_center_xy
    core_scale = max(20.0, float(proposal.core_scale))
    preferred = set(proposal.focus_track_ids) | set(proposal.member_track_ids)
    rows: List[Tuple[float, int, Dict[str, Any], Optional[AnomalyResult], float]] = []
    occupancy = 0.0

    for track_id, observation in packet.observations.items():
        bbox = observation.get("bbox_xyxy")
        if bbox is None:
            continue
        cx, cy, scale = _bbox_center_scale(bbox)
        normalized_distance = math.hypot(cx - core_x, cy - core_y) / max(
            20.0,
            0.5 * (core_scale + scale),
        )
        affinity = 1.0 / (1.0 + normalized_distance * normalized_distance)
        occupancy += affinity

        result = packet.results.get(int(track_id))
        stop_score = 0.0
        staticness = 0.0
        position_stability = 0.0
        recent_speed = 0.0
        speed_noise = 0.0
        decel_score = 0.0
        if result is not None:
            stop_score = max(
                float(result.stop_score),
                float(result.debug.get("stop_semantic_score", 0.0)),
                float(result.debug.get("persistent_stop_score", 0.0)),
            )
            staticness = clip01(float(result.debug.get("staticness", 0.0)))
            position_stability = clip01(
                float(result.debug.get("position_stability", 0.0))
            )
            recent_speed = max(0.0, float(result.debug.get("recent_speed", 0.0)))
            speed_noise = max(
                EPS,
                float(result.debug.get("target_speed_noise", 0.0)),
                float(result.debug.get("motion_resolution", 0.0)),
            )
            decel_score = max(0.0, float(result.decel_score))

        stop_probability = stop_score / (1.0 + stop_score)
        latent_stop = max(
            stop_probability,
            math.sqrt(max(0.0, staticness * position_stability)),
        )
        motion_signal = recent_speed / max(recent_speed + speed_noise, EPS)
        identity_support = 1.0 if int(track_id) in preferred else 0.0
        relevance = affinity * (1.0 + max(latent_stop, identity_support))
        rows.append((float(relevance), int(track_id), observation, result, affinity))

    if not rows:
        return {
            "center_x": float(core_x),
            "center_y": float(core_y),
            "scale": float(core_scale),
            "affinity": 0.0,
            "latent_stop": 0.0,
            "motion_signal": 0.0,
            "decel_signal": 0.0,
            "occupancy": 0.0,
            "track_id": -1.0,
        }

    rows.sort(key=lambda item: item[0], reverse=True)
    _, track_id, observation, result, affinity = rows[0]
    cx, cy, scale = _bbox_center_scale(observation["bbox_xyxy"])
    if result is None:
        latent_stop = 0.0
        motion_signal = 0.0
        decel_signal = 0.0
    else:
        stop_score = max(
            float(result.stop_score),
            float(result.debug.get("stop_semantic_score", 0.0)),
            float(result.debug.get("persistent_stop_score", 0.0)),
        )
        staticness = clip01(float(result.debug.get("staticness", 0.0)))
        position_stability = clip01(
            float(result.debug.get("position_stability", 0.0))
        )
        latent_stop = max(
            stop_score / (1.0 + stop_score),
            math.sqrt(max(0.0, staticness * position_stability)),
        )
        recent_speed = max(0.0, float(result.debug.get("recent_speed", 0.0)))
        speed_noise = max(
            EPS,
            float(result.debug.get("target_speed_noise", 0.0)),
            float(result.debug.get("motion_resolution", 0.0)),
        )
        motion_signal = recent_speed / max(recent_speed + speed_noise, EPS)
        decel_signal = max(0.0, float(result.decel_score))
        decel_signal = decel_signal / (1.0 + decel_signal)

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "scale": float(scale),
        "affinity": float(affinity),
        "latent_stop": float(latent_stop),
        "motion_signal": float(motion_signal),
        "decel_signal": float(decel_signal),
        "occupancy": float(occupancy),
        "track_id": float(track_id),
    }


def _sample_stop_audit_indices(
    timeline: Sequence[CropFramePacket],
    span_indices: Sequence[int],
) -> List[int]:


    if not span_indices:
        return []
    start_time = timeline[span_indices[0]].time_sec
    end_time = timeline[span_indices[-1]].time_sec
    target_times: List[float] = []
    current = float(start_time)
    while current < end_time - 0.5 * _STOP_AUDIT_SAMPLE_SECONDS:
        target_times.append(current)
        current += _STOP_AUDIT_SAMPLE_SECONDS
    target_times.append(float(end_time))
    selected: List[int] = []
    for target_time in target_times:
        index = min(
            span_indices,
            key=lambda item: abs(timeline[item].time_sec - target_time),
        )
        if index not in selected:
            selected.append(int(index))
    return selected


def _select_stop_alarm_keyframes(
    timeline: Sequence[CropFramePacket],
    span_indices: Sequence[int],
    alarm_indices: Sequence[int],
    peak_packet_index: int,
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:


    if not span_indices:
        return [], {}, {}
    alarm_set = sorted(set(int(v) for v in alarm_indices if int(v) in set(span_indices)))
    first_alarm = alarm_set[0] if alarm_set else int(peak_packet_index)
    last_alarm = alarm_set[-1] if alarm_set else int(peak_packet_index)
    peak_index = min(span_indices, key=lambda i: abs(i - int(peak_packet_index)))
    pre_alarm = max(span_indices[0], first_alarm - 1)
    mid_alarm = alarm_set[len(alarm_set) // 2] if alarm_set else peak_index
    post_target = timeline[peak_index].time_sec + 0.8
    post_peak = min(span_indices, key=lambda i: abs(timeline[i].time_sec - post_target))
    end_index = span_indices[-1]
    candidates = [
        ("pre_alarm", pre_alarm),
        ("first_alarm", first_alarm),
        ("alarm_build", mid_alarm),
        ("stop_peak", peak_index),
        ("post_peak", post_peak),
        ("end", end_index),
    ]
    priority = {"stop_peak": 6, "first_alarm": 5, "alarm_build": 4, "post_peak": 3, "pre_alarm": 2, "end": 1}
    role_by_index: Dict[int, str] = {}
    for role, index in candidates:
        old = role_by_index.get(index)
        if old is None or priority[role] > priority.get(old, 0):
            role_by_index[int(index)] = role
    selected = sorted(role_by_index)[:_MAX_STOP_ALARM_KEYFRAMES]
    roles = {index: role_by_index[index] for index in selected}
    phases = {role: int(index) for role, index in candidates}
    return selected, roles, phases


def _select_stop_cause_keyframes(
    timeline: Sequence[CropFramePacket],
    span_indices: Sequence[int],
    proposal: RegionalProposal,
    peak_packet_index: int,
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:


    if not span_indices:
        return [], {}, {}
    peak_index = min(span_indices, key=lambda i: abs(i - int(peak_packet_index)))
    alarm_set = sorted(
        int(v) for v in proposal.alarm_packet_indices if int(v) in set(span_indices)
    )
    first_alarm = alarm_set[0] if alarm_set else peak_index
    history_indices = [i for i in span_indices if i <= first_alarm]
    if not history_indices:
        history_indices = [span_indices[0]]
    start_time = timeline[history_indices[0]].time_sec
    first_alarm_time = timeline[first_alarm].time_sec
    duration = max(0.0, first_alarm_time - start_time)
    history_budget = max(2, _MAX_STOP_CAUSE_KEYFRAMES - 2)
    desired_count = min(
        history_budget,
        max(2, int(math.ceil(duration / _STOP_CAUSE_SAMPLE_SECONDS)) + 1),
    )
    target_times = np.linspace(start_time, first_alarm_time, num=desired_count)
    selected: List[int] = []
    for target_time in target_times.tolist():
        index = min(history_indices, key=lambda i: abs(timeline[i].time_sec - target_time))
        if index not in selected:
            selected.append(int(index))
    for index in (first_alarm, peak_index):
        if index not in selected:
            selected.append(int(index))
    selected = sorted(selected)
    if len(selected) > _MAX_STOP_CAUSE_KEYFRAMES:
        keep = np.linspace(0, len(selected) - 1, num=_MAX_STOP_CAUSE_KEYFRAMES).round().astype(int)
        selected = [selected[int(i)] for i in sorted(set(keep.tolist()))]
    roles: Dict[int, str] = {}
    for order, index in enumerate(selected):
        if index == peak_index:
            roles[index] = "stop_peak"
        elif index == first_alarm:
            roles[index] = "first_stop_alarm"
        elif order == 0:
            roles[index] = "history_start"
        elif order == len(selected) - 1:
            roles[index] = "pre_stop"
        else:
            roles[index] = "history_scan"
    phases = {
        "history_start": int(selected[0]),
        "first_stop_alarm": int(first_alarm),
        "stop_peak": int(peak_index),
    }
    return selected, roles, phases


def _select_stop_retrospective_keyframes(
    timeline: Sequence[CropFramePacket],
    span_indices: Sequence[int],
    proposal: RegionalProposal,
    peak_packet_index: int,
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:


    if not span_indices:
        return [], {}, {}

    peak_index = min(
        span_indices,
        key=lambda index: abs(index - int(peak_packet_index)),
    )
    peak_local = span_indices.index(peak_index)
    alarm_set = set(int(v) for v in proposal.alarm_packet_indices)
    alarm_locals = [
        local for local, index in enumerate(span_indices) if int(index) in alarm_set
    ]
    first_alarm_local = min(alarm_locals) if alarm_locals else peak_local

    states = [
        _stop_hotspot_packet_state(timeline[index], proposal)
        for index in span_indices
    ]

    transition_scores = np.zeros(len(span_indices), dtype=float)
    search_end = max(1, min(first_alarm_local, peak_local))
    for local in range(1, search_end + 1):
        previous = states[local - 1]
        current = states[local]
        stop_rise = max(0.0, current["latent_stop"] - previous["latent_stop"])
        motion_drop = max(0.0, previous["motion_signal"] - current["motion_signal"])
        decel = max(previous["decel_signal"], current["decel_signal"])
        center_shift = math.hypot(
            current["center_x"] - previous["center_x"],
            current["center_y"] - previous["center_y"],
        ) / max(20.0, 0.5 * (current["scale"] + previous["scale"]))
        center_shift = center_shift / (1.0 + center_shift)
        occupancy_change = abs(current["occupancy"] - previous["occupancy"])
        occupancy_change = occupancy_change / (1.0 + occupancy_change)
        physical_change = math.sqrt(
            stop_rise * stop_rise
            + motion_drop * motion_drop
            + decel * decel
            + 0.25 * center_shift * center_shift
            + 0.25 * occupancy_change * occupancy_change
        )
        transition_scores[local] = physical_change * math.sqrt(
            max(EPS, max(previous["affinity"], current["affinity"]))
        )

    cause_local = (
        int(np.argmax(transition_scores[1 : search_end + 1])) + 1
        if search_end >= 1
        else peak_local
    )
    cause_before_local = max(0, cause_local - 1)

    cause_time = timeline[span_indices[cause_local]].time_sec
    approach_time = cause_time - 2.0
    approach_local = min(
        range(0, cause_local + 1),
        key=lambda local: abs(timeline[span_indices[local]].time_sec - approach_time),
    )
    cause_after_time = cause_time + 1.0
    cause_after_local = min(
        range(cause_local, len(span_indices)),
        key=lambda local: abs(timeline[span_indices[local]].time_sec - cause_after_time),
    )
    post_peak_local = min(
        range(peak_local, len(span_indices)),
        key=lambda local: abs(
            timeline[span_indices[local]].time_sec
            - (timeline[peak_index].time_sec + min(2.0, max(0.0, timeline[span_indices[-1]].time_sec - timeline[peak_index].time_sec)))
        ),
    )

    phase_candidates = [
        ("early_context", 0),
        ("approach", approach_local),
        ("cause_before", cause_before_local),
        ("cause", cause_local),
        ("cause_after", cause_after_local),
        ("first_stop_alarm", first_alarm_local),
        ("stop_peak", peak_local),
        ("post_stop", post_peak_local),
    ]

    role_by_local: Dict[int, str] = {}
    priority = {
        "cause": 8,
        "cause_before": 7,
        "cause_after": 6,
        "stop_peak": 5,
        "first_stop_alarm": 4,
        "approach": 3,
        "early_context": 2,
        "post_stop": 1,
    }
    for role, local in phase_candidates:
        current_role = role_by_local.get(local)
        if current_role is None or priority[role] > priority.get(current_role, 0):
            role_by_local[int(local)] = str(role)

    selected_locals = sorted(role_by_local)
    if len(selected_locals) > _MAX_STOP_VLM_KEYFRAMES:
        selected_locals = sorted(
            sorted(
                selected_locals,
                key=lambda local: (
                    priority.get(role_by_local[local], 0),
                    states[local]["latent_stop"],
                ),
                reverse=True,
            )[:_MAX_STOP_VLM_KEYFRAMES]
        )

    selected_indices = [int(span_indices[local]) for local in selected_locals]
    roles = {
        int(span_indices[local]): str(role_by_local[local])
        for local in selected_locals
    }
    phase_indices = {
        str(role): int(span_indices[local])
        for role, local in phase_candidates
    }
    phase_indices["audit_start"] = int(span_indices[0])
    phase_indices["audit_end"] = int(span_indices[-1])
    return selected_indices, roles, phase_indices


def _select_persistent_alarm_keyframes(
    timeline: Sequence[CropFramePacket],
    span_indices: Sequence[int],
    proposal: RegionalProposal,
    peak_packet_index: int,
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:


    if not span_indices:
        return [], {}, {}

    peak_local = min(
        range(len(span_indices)),
        key=lambda local: abs(span_indices[local] - int(peak_packet_index)),
    )
    alarm_set = set(int(v) for v in proposal.alarm_packet_indices)
    alarm_locals = [
        local for local, index in enumerate(span_indices) if int(index) in alarm_set
    ] or [peak_local]
    first_alarm_local = min(alarm_locals)
    last_alarm_local = max(alarm_locals)

    first_time = timeline[span_indices[first_alarm_local]].time_sec
    early_time = first_time - 2.0
    early_local = min(
        range(0, first_alarm_local + 1),
        key=lambda local: abs(timeline[span_indices[local]].time_sec - early_time),
    )
    pre_alarm_local = max(0, first_alarm_local - 1)


    build_end = max(first_alarm_local, peak_local)
    build_candidates = list(range(first_alarm_local, build_end + 1))
    if build_candidates:
        start_t = timeline[span_indices[first_alarm_local]].time_sec
        end_t = timeline[span_indices[build_end]].time_sec
        build_early_t = start_t + (end_t - start_t) / 3.0
        build_late_t = start_t + 2.0 * (end_t - start_t) / 3.0
        build_early_local = min(
            build_candidates,
            key=lambda local: abs(timeline[span_indices[local]].time_sec - build_early_t),
        )
        build_late_local = min(
            build_candidates,
            key=lambda local: abs(timeline[span_indices[local]].time_sec - build_late_t),
        )
    else:
        build_early_local = build_late_local = peak_local

    post_time = timeline[span_indices[last_alarm_local]].time_sec + 1.0
    post_local = min(
        range(last_alarm_local, len(span_indices)),
        key=lambda local: abs(timeline[span_indices[local]].time_sec - post_time),
    )
    end_local = len(span_indices) - 1

    phase_candidates = [
        ("early_context", early_local),
        ("pre_alarm", pre_alarm_local),
        ("onset", first_alarm_local),
        ("build_up_early", build_early_local),
        ("build_up_late", build_late_local),
        ("peak", peak_local),
        ("post_alarm", post_local),
        ("end", end_local),
    ]
    priority = {
        "peak": 8,
        "build_up_late": 7,
        "build_up_early": 6,
        "onset": 5,
        "post_alarm": 4,
        "pre_alarm": 3,
        "early_context": 2,
        "end": 1,
    }
    role_by_local: Dict[int, str] = {}
    for role, local in phase_candidates:
        current = role_by_local.get(local)
        if current is None or priority[role] > priority.get(current, 0):
            role_by_local[int(local)] = str(role)

    selected_locals = sorted(role_by_local)
    if len(selected_locals) > _MAX_PERSISTENT_VLM_KEYFRAMES:
        selected_locals = sorted(
            sorted(
                selected_locals,
                key=lambda local: (priority.get(role_by_local[local], 0), -abs(local-peak_local)),
                reverse=True,
            )[:_MAX_PERSISTENT_VLM_KEYFRAMES]
        )
    selected_indices = [int(span_indices[local]) for local in selected_locals]
    roles = {int(span_indices[local]): str(role_by_local[local]) for local in selected_locals}
    phases = {str(role): int(span_indices[local]) for role, local in phase_candidates}
    return selected_indices, roles, phases


def _select_forward_outcome_keyframes(
    timeline: Sequence[CropFramePacket],
    span_indices: Sequence[int],
    anchor_index: int,
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:


    if not span_indices:
        return [], {}, {}
    anchor_time = timeline[anchor_index].time_sec
    targets = [
        ("forward_pre", anchor_time - 0.35),
        ("forward_alarm", anchor_time),
        ("forward_early", anchor_time + 0.55),
        ("forward_mid", anchor_time + 1.10),
        ("forward_late", anchor_time + 1.70),
        ("forward_end", anchor_time + _FORWARD_OUTCOME_SECONDS),
    ]
    role_by_index: Dict[int, str] = {}
    phases: Dict[str, int] = {}
    for role, target_time in targets:
        index = min(
            span_indices,
            key=lambda value: abs(timeline[value].time_sec - target_time),
        )
        phases[role] = int(index)
        role_by_index.setdefault(int(index), role)
    selected = sorted(role_by_index)[:_MAX_FORWARD_OUTCOME_KEYFRAMES]
    return selected, {index: role_by_index[index] for index in selected}, phases


def _estimate_forward_motion(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    anchor_index: int,
) -> Tuple[Tuple[float, float], float, float]:


    anchor_time = timeline[anchor_index].time_sec
    vectors: List[Tuple[float, float, float, float]] = []
    focus_ids = proposal.focus_track_ids or [proposal.representative_track_id]
    for track_id in focus_ids:
        rows: List[Tuple[float, float, float, float]] = []
        for packet in timeline[: anchor_index + 1]:
            if packet.time_sec < anchor_time - 2.5:
                continue
            observation = packet.observations.get(int(track_id))
            if observation is None:
                continue
            cx, cy, scale = _bbox_center_scale(observation["bbox_xyxy"])
            rows.append((packet.time_sec, cx, cy, scale))
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        dt = max(EPS, last[0] - first[0])
        dx, dy = last[1] - first[1], last[2] - first[2]
        distance = math.hypot(dx, dy)
        scale = max(20.0, float(np.median([row[3] for row in rows])))
        if distance < 0.20 * scale:
            continue
        relevance = max(0.1, float(proposal.track_relevance.get(int(track_id), 0.5)))
        vectors.append((dx / dt, dy / dt, relevance * min(2.5, dt), scale))


    alarm_rows = sorted(
        [sample for sample in proposal.alarm_samples if sample.packet_index <= anchor_index],
        key=lambda sample: sample.packet_index,
    )
    if len(alarm_rows) >= 2:
        first, last = alarm_rows[max(0, len(alarm_rows) - 5)], alarm_rows[-1]
        dt = max(
            EPS,
            timeline[last.packet_index].time_sec - timeline[first.packet_index].time_sec,
        )
        dx = last.center_xy[0] - first.center_xy[0]
        dy = last.center_xy[1] - first.center_xy[1]
        if math.hypot(dx, dy) >= 0.20 * max(20.0, proposal.core_scale):
            vectors.append((dx / dt, dy / dt, 0.75, proposal.core_scale))

    observed_direction: Optional[Tuple[float, float]] = None
    observed_speed = 0.0
    consistency = 0.0
    if vectors:
        weighted_x = sum(row[0] * row[2] for row in vectors)
        weighted_y = sum(row[1] * row[2] for row in vectors)
        weight_sum = sum(row[2] for row in vectors)
        mean_x, mean_y = weighted_x / max(weight_sum, EPS), weighted_y / max(weight_sum, EPS)
        observed_speed = math.hypot(mean_x, mean_y)
        if observed_speed > EPS:
            observed_direction = (mean_x / observed_speed, mean_y / observed_speed)
            unit_sum_x = 0.0
            unit_sum_y = 0.0
            for vx, vy, weight, _ in vectors:
                norm = math.hypot(vx, vy)
                if norm <= EPS:
                    continue
                unit_sum_x += weight * vx / norm
                unit_sum_y += weight * vy / norm
            consistency = clip01(math.hypot(unit_sum_x, unit_sum_y) / max(weight_sum, EPS))

    packet = timeline[anchor_index]
    _, result = _best_region_result(packet, proposal.member_track_ids, proposal.core_roi_xyxy)
    model_direction: Optional[Tuple[float, float]] = None
    model_speed = 0.0
    if result is not None and result.road_axis.ready:
        axis_x, axis_y = safe_unit(result.road_axis.x, result.road_axis.y)
        perp_x, perp_y = -axis_y, axis_x
        vx = float(result.v_parallel) * axis_x + float(result.v_perpendicular) * perp_x
        vy = float(result.v_parallel) * axis_y + float(result.v_perpendicular) * perp_y
        norm = math.hypot(vx, vy)
        if norm > EPS:
            model_direction = (vx / norm, vy / norm)
            model_speed = norm * max(20.0, proposal.core_scale) * _packet_fps(timeline)

    if observed_direction is not None:
        direction = observed_direction
        speed = observed_speed
        if model_direction is not None:
            alignment = direction[0] * model_direction[0] + direction[1] * model_direction[1]
            if alignment > 0.25:
                bx = 0.75 * direction[0] + 0.25 * model_direction[0]
                by = 0.75 * direction[1] + 0.25 * model_direction[1]
                direction = safe_unit(bx, by)
                speed = max(speed, 0.5 * model_speed)
        return direction, max(0.0, speed), max(0.35, consistency)
    if model_direction is not None:
        return model_direction, max(0.0, model_speed), 0.35
    return (0.0, 0.0), 0.0, 0.0


def _forward_outcome_roi(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    keyframe_indices: Sequence[int],
    anchor_index: int,
    direction: Tuple[float, float],
    speed_px_s: float,
    horizon_seconds: float,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:


    anchor_packet = timeline[anchor_index]
    width, height = anchor_packet.frame_width, anchor_packet.frame_height
    dir_x, dir_y = safe_unit(direction[0], direction[1])
    if abs(dir_x) + abs(dir_y) <= EPS:
        return tuple(int(round(v)) for v in proposal.core_roi_xyxy), proposal.core_center_xy
    perp_x, perp_y = -dir_y, dir_x
    scale = max(20.0, float(proposal.core_scale))
    anchor_x, anchor_y = proposal.core_center_xy
    predicted = speed_px_s * max(0.0, horizon_seconds)
    forward_distance = float(np.clip(predicted, 1.5 * scale, 7.0 * scale))
    endpoint = (anchor_x + forward_distance * dir_x, anchor_y + forward_distance * dir_y)

    points: List[Tuple[float, float]] = [(anchor_x, anchor_y), endpoint]
    boxes: List[Tuple[float, float, float, float]] = []
    focus_ids = set(proposal.focus_track_ids + proposal.member_track_ids)
    anchor_time = anchor_packet.time_sec
    for index in keyframe_indices:
        packet = timeline[index]
        dt = max(0.0, packet.time_sec - anchor_time)
        expected = (
            anchor_x + min(forward_distance, speed_px_s * dt) * dir_x,
            anchor_y + min(forward_distance, speed_px_s * dt) * dir_y,
        )
        best: Optional[Tuple[float, Tuple[float, float, float, float]]] = None
        for track_id, observation in packet.observations.items():
            bbox = tuple(float(v) for v in observation["bbox_xyxy"])
            cx, cy, box_scale = _bbox_center_scale(bbox)
            distance = math.hypot(cx - expected[0], cy - expected[1]) / max(20.0, 0.5 * (scale + box_scale))
            identity_bonus = 0.35 if int(track_id) in focus_ids else 0.0
            score = distance - identity_bonus
            if best is None or score < best[0]:
                best = (score, bbox)
        if best is not None and best[0] <= 2.2:
            boxes.append(best[1])
            points.append(_bbox_center_scale(best[1])[:2])
        else:
            points.append(expected)

    relative_parallel = [
        (x - anchor_x) * dir_x + (y - anchor_y) * dir_y
        for x, y in points
    ]
    relative_perp = [
        (x - anchor_x) * perp_x + (y - anchor_y) * perp_y
        for x, y in points
    ]
    min_parallel = min(-0.8 * scale, min(relative_parallel) - 0.5 * scale)
    max_parallel = max(forward_distance + 0.8 * scale, max(relative_parallel) + 0.5 * scale)
    half_perp = max(1.35 * scale, max(abs(v) for v in relative_perp) + 0.75 * scale)

    corners = []
    for along in (min_parallel, max_parallel):
        for across in (-half_perp, half_perp):
            corners.append(
                (
                    anchor_x + along * dir_x + across * perp_x,
                    anchor_y + along * dir_y + across * perp_y,
                )
            )
    min_x = min(point[0] for point in corners)
    min_y = min(point[1] for point in corners)
    max_x = max(point[0] for point in corners)
    max_y = max(point[1] for point in corners)
    for box in boxes:
        min_x = min(min_x, box[0] - 0.35 * scale)
        min_y = min(min_y, box[1] - 0.35 * scale)
        max_x = max(max_x, box[2] + 0.35 * scale)
        max_y = max(max_y, box[3] + 0.35 * scale)
    roi = (
        max(0, int(math.floor(min_x))),
        max(0, int(math.floor(min_y))),
        min(width, int(math.ceil(max_x))),
        min(height, int(math.ceil(max_y))),
    )
    if roi[2] - roi[0] < 2 or roi[3] - roi[1] < 2:
        roi = tuple(int(round(v)) for v in proposal.core_roi_xyxy)
    return roi, endpoint


def _select_regional_keyframes(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    span_indices: Sequence[int],
    roi_xyxy: Sequence[float],
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:


    if not span_indices:
        index = int(proposal.peak_packet_index)
        return [index], {index: "peak"}, {"peak": index}

    count = min(
        len(span_indices),
        _MAX_KEYFRAMES,
        max(_MIN_KEYFRAMES, int(math.ceil(math.sqrt(len(span_indices))))),
    )
    if len(span_indices) <= count:
        indices = list(span_indices)
        roles = {
            index: (
                "peak"
                if index == proposal.peak_packet_index
                else "start"
                if order == 0
                else "end"
                if order == len(indices) - 1
                else "transition"
            )
            for order, index in enumerate(indices)
        }
        return indices, roles, {"peak": int(proposal.peak_packet_index)}

    states = [_regional_state(timeline[index], proposal, roi_xyxy) for index in span_indices]
    times = np.asarray([timeline[index].time_sec for index in span_indices], dtype=float)
    raw = np.asarray(
        [
            [
                state["center_x"],
                state["center_y"],
                math.log(max(1.0, state["scale"])),
                state["score"],
                *state["branches"],
                state["contact_affinity"],
            ]
            for state in states
        ],
        dtype=float,
    )

    def normalize_column(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values, dtype=float)
        median = float(np.median(finite))
        mad = float(1.4826 * np.median(np.abs(finite - median)))
        total_span = float(np.max(finite) - np.min(finite))
        scale = max(mad, total_span / math.sqrt(max(1.0, float(len(finite)))), EPS)
        return (values - median) / scale

    normalized_state = np.column_stack(
        [normalize_column(raw[:, column]) for column in range(raw.shape[1])]
    )
    consecutive_change = np.linalg.norm(np.diff(normalized_state, axis=0), axis=1)

    alarm_locals = [
        local
        for local, packet_index in enumerate(span_indices)
        if packet_index in set(proposal.alarm_packet_indices)
    ]
    first_alarm_local = min(alarm_locals) if alarm_locals else min(
        range(len(span_indices)),
        key=lambda local: abs(span_indices[local] - proposal.peak_packet_index),
    )
    peak_local = min(
        range(len(span_indices)),
        key=lambda local: abs(span_indices[local] - proposal.peak_packet_index),
    )

    branch_matrix = raw[:, 4:9]
    branch_strength = np.max(branch_matrix, axis=1)
    score_support = np.asarray(
        [clip01(state["score"] / max(proposal.peak_score, EPS)) for state in states],
        dtype=float,
    )
    contact = np.asarray([state["contact_affinity"] for state in states], dtype=float)


    precursor_end = max(first_alarm_local, peak_local)
    precursor_candidates = list(range(first_alarm_local, precursor_end + 1))
    precursor_local = max(
        precursor_candidates,
        key=lambda local: (
            states[local]["score"],
            branch_strength[local],
        ),
    )


    interaction_metric = np.zeros(len(span_indices), dtype=float)
    for local in range(1, len(span_indices)):
        support = max(
            score_support[local - 1],
            score_support[local],
            clip01(branch_strength[local - 1] / max(proposal.peak_score, EPS)),
            clip01(branch_strength[local] / max(proposal.peak_score, EPS)),
            contact[local - 1],
            contact[local],
        )
        interaction_metric[local] = consecutive_change[local - 1] * math.sqrt(
            max(EPS, support)
        )
    search_start = max(1, precursor_local + 1)
    interaction_after_local = (
        int(search_start + np.argmax(interaction_metric[search_start:]))
        if search_start < len(span_indices)
        else peak_local
    )
    interaction_before_local = max(precursor_local, interaction_after_local - 1)


    pre_context_local = max(0, first_alarm_local - 1)


    consequence_local = interaction_after_local
    consequence_value = -1.0
    for local in range(interaction_after_local, len(span_indices)):
        lateral, _, decel, stop, impact = states[local]["branches"]
        consequence_signal = max(lateral, decel, stop, impact)
        offset_reliability = _continuous_sample_reliability(
            local - interaction_after_local + 1
        )
        value = consequence_signal * math.sqrt(max(EPS, offset_reliability))
        if value > consequence_value:
            consequence_value = value
            consequence_local = local

    end_local = len(span_indices) - 1
    phase_candidates = [
        ("pre_context", pre_context_local),
        ("precursor", precursor_local),
        ("interaction_before", interaction_before_local),
        ("interaction", interaction_after_local),
        ("consequence", consequence_local),
        ("end", end_local),
    ]

    selected_locals: List[int] = []
    role_by_local: Dict[int, str] = {}
    for role, local in phase_candidates:
        if local not in role_by_local:
            selected_locals.append(local)
            role_by_local[local] = role
        elif role == "interaction":
            role_by_local[local] = role


    if peak_local not in role_by_local and len(selected_locals) < count:
        selected_locals.append(peak_local)
        role_by_local[peak_local] = "peak"
    total_span = max(EPS, float(times[-1] - times[0]))
    novelty_features = np.column_stack(
        [
            (times - times[0]) / total_span,
            normalize_column(raw[:, 3]),
            normalize_column(raw[:, 0]),
            normalize_column(raw[:, 1]),
            normalize_column(raw[:, 8]),
            contact,
        ]
    )
    while len(selected_locals) < count:
        best_local: Optional[int] = None
        best_novelty = -1.0
        for candidate in range(len(span_indices)):
            if candidate in role_by_local:
                continue
            novelty = min(
                float(np.linalg.norm(novelty_features[candidate] - novelty_features[item]))
                for item in selected_locals
            )
            if novelty > best_novelty:
                best_novelty = novelty
                best_local = candidate
        if best_local is None:
            break
        selected_locals.append(best_local)
        role_by_local[best_local] = "transition"


    if len(selected_locals) > count:
        role_priority = {
            "interaction": 6,
            "interaction_before": 5,
            "consequence": 4,
            "precursor": 3,
            "pre_context": 2,
            "peak": 2,
            "end": 1,
            "transition": 0,
        }
        selected_locals = sorted(
            selected_locals,
            key=lambda local: (
                role_priority.get(role_by_local[local], 0),
                states[local]["score"],
            ),
            reverse=True,
        )[:count]

    selected_locals = sorted(set(selected_locals))
    selected_indices = [span_indices[local] for local in selected_locals]
    roles = {
        span_indices[local]: (
            "peak"
            if span_indices[local] == proposal.peak_packet_index
            and role_by_local.get(local) not in {"interaction", "interaction_before", "consequence"}
            else role_by_local.get(local, "transition")
        )
        for local in selected_locals
    }
    phase_indices = {
        role: int(span_indices[local])
        for role, local in phase_candidates
    }
    phase_indices["peak"] = int(span_indices[peak_local])
    return selected_indices, roles, phase_indices


def _select_relevant_neighbors_for_region(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    span_indices: Sequence[int],
) -> Tuple[List[int], Dict[int, float]]:
    evidence: Dict[int, List[float]] = {}
    member_set = set(proposal.member_track_ids)
    source_track_ids = proposal.focus_track_ids or [proposal.representative_track_id]
    for index in span_indices:
        packet = timeline[index]
        for source_id in source_track_ids:
            result = packet.results.get(source_id)
            context_weight_map: Dict[int, float] = {}
            if result is not None:
                raw_map = result.debug.get("context_track_weights", {})
                if isinstance(raw_map, dict):
                    for key, value in raw_map.items():
                        try:
                            context_weight_map[int(key)] = max(0.0, float(value))
                        except (TypeError, ValueError):
                            continue
            for neighbor_id in packet.neighbor_map.get(source_id, []):
                if neighbor_id in member_set or neighbor_id not in packet.observations:
                    continue
                distance = packet.distance_map.get((source_id, neighbor_id), float("inf"))
                proximity = 0.0 if not math.isfinite(distance) else 1.0 / (1.0 + max(0.0, distance))
                weight = max(proximity, context_weight_map.get(neighbor_id, 0.0))
                if weight > 0.0:
                    evidence.setdefault(int(neighbor_id), []).append(float(weight))
            for neighbor_id, weight in context_weight_map.items():
                if neighbor_id not in member_set and neighbor_id in packet.observations:
                    evidence.setdefault(int(neighbor_id), []).append(float(weight))

    if not evidence:
        return [], {}


    aggregate = {
        track: float(math.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))
        for track, values in evidence.items()
        if values
    }
    sorted_items = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
    weights = np.asarray([max(item[1], EPS) for item in sorted_items], dtype=float)
    effective_count = _effective_sample_size(weights)
    keep_count = min(
        len(sorted_items),
        max(1, int(math.ceil(math.sqrt(max(1.0, effective_count))))),
    )
    selected = sorted_items[:keep_count]
    return [int(item[0]) for item in selected], {int(k): float(v) for k, v in selected}


def _hotspot_frame_boxes(
    packet: CropFramePacket,
    core_center_xy: Tuple[float, float],
    core_scale: float,
    preferred_track_ids: Sequence[int],
    max_boxes: int = 3,
) -> List[Tuple[float, float, float, float]]:


    preferred = set(int(v) for v in preferred_track_ids)
    rows: List[Tuple[float, Tuple[float, float, float, float]]] = []
    cx0, cy0 = core_center_xy
    scale0 = max(20.0, float(core_scale))
    for track_id, observation in packet.observations.items():
        bbox = tuple(float(v) for v in observation["bbox_xyxy"])
        cx, cy, scale = _bbox_center_scale(bbox)
        normalized_distance = math.hypot(cx - cx0, cy - cy0) / max(20.0, 0.5 * (scale0 + scale))
        proximity = 1.0 / (1.0 + normalized_distance * normalized_distance)
        result = packet.results.get(int(track_id))
        stop_support = 0.0
        if result is not None:
            stop_support = max(
                float(result.stop_score),
                float(result.debug.get("stop_semantic_score", 0.0)),
            )
            stop_support = clip01(stop_support / max(1.0, float(result.score), EPS))
        identity_support = 1.0 if int(track_id) in preferred else 0.0
        relevance = proximity * (1.0 + max(stop_support, identity_support))
        rows.append((float(relevance), bbox))
    rows.sort(key=lambda item: item[0], reverse=True)
    if not rows:
        return []
    weights = np.asarray([max(EPS, value) for value, _ in rows], dtype=float)
    effective = _effective_sample_size(weights)
    keep = min(
        len(rows),
        max(1, min(int(max_boxes), int(math.ceil(math.sqrt(max(1.0, effective)))) + 1)),
    )
    return [bbox for _, bbox in rows[:keep]]


def _road_aligned_hotspot_roi(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    parallel_scale: float,
    perpendicular_scale: float,
) -> Tuple[int, int, int, int]:


    peak_packet = timeline[proposal.peak_packet_index]
    width, height = peak_packet.frame_width, peak_packet.frame_height
    cx, cy = proposal.core_center_xy
    scale = max(20.0, float(proposal.core_scale))
    peak_result = peak_packet.results.get(proposal.representative_track_id)
    if peak_result is not None and peak_result.road_axis.ready:
        ax, ay = safe_unit(peak_result.road_axis.x, peak_result.road_axis.y)
    else:
        ax, ay = 1.0, 0.0
    px, py = -ay, ax
    hp = parallel_scale * scale
    hq = perpendicular_scale * scale
    corners = [
        (cx + sp * hp * ax + sq * hq * px, cy + sp * hp * ay + sq * hq * py)
        for sp in (-1.0, 1.0)
        for sq in (-1.0, 1.0)
    ]
    x1 = max(0, int(math.floor(min(v[0] for v in corners))))
    y1 = max(0, int(math.floor(min(v[1] for v in corners))))
    x2 = min(width, int(math.ceil(max(v[0] for v in corners))))
    y2 = min(height, int(math.ceil(max(v[1] for v in corners))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _stop_alarm_context_roi(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    keyframe_indices: Sequence[int],
) -> Tuple[int, int, int, int]:

    return _road_aligned_hotspot_roi(timeline, proposal, 1.65, 1.05)


def _stop_cause_context_roi(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    keyframe_indices: Sequence[int],
) -> Tuple[int, int, int, int]:

    return _road_aligned_hotspot_roi(timeline, proposal, 2.65, 1.45)


def _adaptive_stop_regional_roi(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    keyframe_indices: Sequence[int],
    neighbor_ids: Sequence[int],
) -> Tuple[int, int, int, int]:


    peak_packet = timeline[proposal.peak_packet_index]
    width, height = peak_packet.frame_width, peak_packet.frame_height
    preferred = list(proposal.focus_track_ids) + list(proposal.member_track_ids)
    boxes: List[Tuple[float, float, float, float]] = []
    centers: List[Tuple[float, float]] = []
    for index in keyframe_indices:
        packet = timeline[index]
        frame_boxes = _hotspot_frame_boxes(
            packet,
            proposal.core_center_xy,
            proposal.core_scale,
            preferred,
            max_boxes=3,
        )
        boxes.extend(frame_boxes)
        centers.extend([_bbox_center_scale(box)[:2] for box in frame_boxes])

    if not boxes:
        boxes = [tuple(float(v) for v in proposal.core_roi_xyxy)]
        centers = [_bbox_center_scale(boxes[0])[:2]]


    peak_result = peak_packet.results.get(proposal.representative_track_id)
    if peak_result is None:
        _, peak_result = _best_region_result(
            peak_packet,
            proposal.member_track_ids,
            proposal.core_roi_xyxy,
        )
    core_x, core_y = proposal.core_center_xy
    core_scale = max(20.0, float(proposal.core_scale))
    if peak_result is not None and peak_result.road_axis.ready:
        axis_x, axis_y = safe_unit(peak_result.road_axis.x, peak_result.road_axis.y)
        perp_x, perp_y = -axis_y, axis_x
        points = np.asarray(centers, dtype=float)
        relative = points - np.asarray([[core_x, core_y]], dtype=float)
        parallel_span = float(np.ptp(relative[:, 0] * axis_x + relative[:, 1] * axis_y)) if len(points) > 1 else 0.0
        perpendicular_span = float(np.ptp(relative[:, 0] * perp_x + relative[:, 1] * perp_y)) if len(points) > 1 else 0.0
        half_parallel = core_scale + math.sqrt(core_scale * max(core_scale, parallel_span))
        half_perpendicular = core_scale + math.sqrt(core_scale * max(core_scale, perpendicular_span))
        corners = []
        for sign_p in (-1.0, 1.0):
            for sign_q in (-1.0, 1.0):
                corners.append(
                    (
                        core_x + sign_p * half_parallel * axis_x + sign_q * half_perpendicular * perp_x,
                        core_y + sign_p * half_parallel * axis_y + sign_q * half_perpendicular * perp_y,
                    )
                )
        min_cx = min(point[0] for point in corners)
        min_cy = min(point[1] for point in corners)
        max_cx = max(point[0] for point in corners)
        max_cy = max(point[1] for point in corners)
        boxes.append((min_cx, min_cy, max_cx, max_cy))

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    spread = max(max_x - min_x, max_y - min_y)
    padding = math.sqrt(core_scale * core_scale + core_scale * max(core_scale, spread))
    x1 = max(0, int(math.floor(min_x - padding)))
    y1 = max(0, int(math.floor(min_y - padding)))
    x2 = min(width, int(math.ceil(max_x + padding)))
    y2 = min(height, int(math.ceil(max_y + padding)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _adaptive_regional_roi(
    timeline: Sequence[CropFramePacket],
    proposal: RegionalProposal,
    span_indices: Sequence[int],
    keyframe_indices: Sequence[int],
    neighbor_ids: Sequence[int],
    lookback_seconds: float,
    future_seconds: float,
) -> Tuple[int, int, int, int]:


    peak_packet = timeline[proposal.peak_packet_index]
    width, height = peak_packet.frame_width, peak_packet.frame_height
    focus_ids = proposal.focus_track_ids or [proposal.representative_track_id]
    member_set = set(proposal.member_track_ids)
    keyframe_set = set(int(v) for v in keyframe_indices)
    boxes: List[Tuple[float, float, float, float]] = []
    target_boxes: List[Tuple[float, float, float, float]] = []
    ground_points: List[Tuple[float, float]] = []
    target_scales: List[float] = []

    for index in span_indices:
        packet = timeline[index]
        active_focus = [
            track_id for track_id in focus_ids if track_id in packet.observations
        ]
        if not active_focus:
            active_focus = [
                track_id
                for track_id in proposal.member_track_ids
                if track_id in packet.observations
                and _bbox_matches_region(
                    packet.observations[track_id]["bbox_xyxy"],
                    proposal.core_roi_xyxy,
                )
            ]
        for track_id in active_focus:
            target = packet.observations.get(track_id)
            if target is None:
                continue
            box = tuple(float(v) for v in target["bbox_xyxy"])
            boxes.append(box)
            target_boxes.append(box)
            x1, y1, x2, y2 = box
            scale = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
            target_scales.append(scale)
            ground_points.append((0.5 * (x1 + x2), y1 + 0.96 * (y2 - y1)))


        if index in keyframe_set:
            for neighbor_id in neighbor_ids:
                if neighbor_id in member_set:
                    continue
                neighbor = packet.observations.get(neighbor_id)
                if neighbor is not None:
                    boxes.append(tuple(float(v) for v in neighbor["bbox_xyxy"]))

    if not target_boxes:
        boxes.append(tuple(float(v) for v in proposal.core_roi_xyxy))
        target_boxes.append(tuple(float(v) for v in proposal.core_roi_xyxy))

    median_scale = float(np.median(target_scales)) if target_scales else max(
        24.0,
        math.sqrt(
            max(
                1.0,
                (proposal.core_roi_xyxy[2] - proposal.core_roi_xyxy[0])
                * (proposal.core_roi_xyxy[3] - proposal.core_roi_xyxy[1]),
            )
        ) / 3.0,
    )
    peak_result = peak_packet.results.get(proposal.representative_track_id)
    peak_observation = peak_packet.observations.get(proposal.representative_track_id)

    if peak_result is not None and peak_result.road_axis.ready:
        axis_x, axis_y = safe_unit(peak_result.road_axis.x, peak_result.road_axis.y)
        perp_x, perp_y = -axis_y, axis_x
        if peak_observation is not None:
            px1, py1, px2, py2 = [float(v) for v in peak_observation["bbox_xyxy"]]
            peak_ground = np.asarray(
                [0.5 * (px1 + px2), py1 + 0.96 * (py2 - py1)],
                dtype=float,
            )
        elif ground_points:
            peak_ground = np.median(np.asarray(ground_points, dtype=float), axis=0)
        else:
            x1, y1, x2, y2 = proposal.core_roi_xyxy
            peak_ground = np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=float)

        moving_reference = max(
            0.0,
            float(peak_result.debug.get("moving_reference", 0.0)),
            abs(float(peak_result.debug.get("mu_parallel", 0.0))),
        )
        current_parallel = float(peak_result.v_parallel)
        current_perpendicular = float(peak_result.v_perpendicular)
        packet_fps = max(1.0, _packet_fps(timeline))
        predicted_backward = (
            moving_reference
            * median_scale
            * max(0.0, lookback_seconds)
            * packet_fps
        )
        predicted_forward = (
            abs(current_parallel)
            * median_scale
            * max(0.0, future_seconds)
            * packet_fps
        )
        predicted_lateral = (
            abs(current_perpendicular)
            * median_scale
            * max(0.0, future_seconds)
            * packet_fps
        )

        observed_parallel_span = median_scale
        observed_perpendicular_span = median_scale
        if ground_points:
            points = np.asarray(ground_points, dtype=float)
            relative = points - peak_ground[None, :]
            observed_parallel_span += float(
                np.ptp(relative[:, 0] * axis_x + relative[:, 1] * axis_y)
            )
            observed_perpendicular_span += float(
                np.ptp(relative[:, 0] * perp_x + relative[:, 1] * perp_y)
            )

        backward_distance = math.sqrt(
            max(0.0, predicted_backward * observed_parallel_span)
        )
        forward_distance = math.sqrt(
            max(0.0, predicted_forward * observed_parallel_span)
        )
        lateral_distance = math.sqrt(
            max(0.0, predicted_lateral * observed_perpendicular_span)
        )
        direction_sign = 1.0 if current_parallel >= 0.0 else -1.0
        backward_point = (
            peak_ground
            - direction_sign * backward_distance * np.asarray([axis_x, axis_y])
        )
        forward_point = (
            peak_ground
            + direction_sign * forward_distance * np.asarray([axis_x, axis_y])
            + lateral_distance * np.asarray([perp_x, perp_y])
        )
        corridor_half_width = median_scale
        for point in (backward_point, peak_ground, forward_point):
            boxes.append(
                (
                    float(point[0] - corridor_half_width),
                    float(point[1] - corridor_half_width),
                    float(point[0] + corridor_half_width),
                    float(point[1] + corridor_half_width),
                )
            )

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)

    if len(ground_points) >= 2:
        points = np.asarray(ground_points, dtype=float)
        path_spread = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    else:
        path_spread = 0.0


    padding = math.sqrt(
        max(median_scale * median_scale, median_scale * median_scale + median_scale * path_spread)
    )
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    x1 = max(0, int(math.floor(min_x)))
    y1 = max(0, int(math.floor(min_y)))
    x2 = min(width, int(math.ceil(max_x)))
    y2 = min(height, int(math.ceil(max_y)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _fixed_stop_alarm_focus_roi(
    timeline: Sequence[CropFramePacket],
    event: AdaptiveEvent,
) -> Tuple[int, int, int, int]:
    return event.roi_xyxy


def _fixed_stop_cause_focus_roi(
    timeline: Sequence[CropFramePacket],
    event: AdaptiveEvent,
) -> Tuple[int, int, int, int]:
    return event.roi_xyxy


def _fixed_stop_focus_roi(
    timeline: Sequence[CropFramePacket],
    event: AdaptiveEvent,
) -> Tuple[int, int, int, int]:


    context_x1, context_y1, context_x2, context_y2 = event.roi_xyxy
    preferred_roles = {
        "cause_before",
        "cause",
        "cause_after",
        "first_stop_alarm",
        "stop_peak",
        "post_stop",
    }
    boxes: List[Tuple[float, float, float, float]] = []
    preferred = list(event.focus_track_ids) + list(event.member_track_ids)
    for index in event.keyframe_indices:
        role = event.keyframe_roles.get(int(index), "")
        if role not in preferred_roles:
            continue
        boxes.extend(
            _hotspot_frame_boxes(
                timeline[index],
                event.core_center_xy,
                event.core_scale,
                preferred,
                max_boxes=2,
            )
        )

    if not boxes:
        boxes = [tuple(float(v) for v in event.core_roi_xyxy)]

    centers = np.asarray(
        [[0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])] for box in boxes],
        dtype=float,
    )
    scales = np.asarray([_bbox_center_scale(box)[2] for box in boxes], dtype=float)
    core_scale = max(20.0, float(event.core_scale), float(np.median(scales)))


    low_x, high_x = np.quantile(centers[:, 0], [0.20, 0.80])
    low_y, high_y = np.quantile(centers[:, 1], [0.20, 0.80])
    center_x = 0.5 * (float(low_x) + float(high_x))
    center_y = 0.5 * (float(low_y) + float(high_y))
    spread_x = max(0.5 * core_scale, float(high_x - low_x))
    spread_y = max(0.5 * core_scale, float(high_y - low_y))


    half_width = 1.05 * core_scale + 0.78 * math.sqrt(
        core_scale * max(core_scale, spread_x)
    )
    half_height = 1.05 * core_scale + 0.78 * math.sqrt(
        core_scale * max(core_scale, spread_y)
    )


    hotspot_x, hotspot_y = event.core_center_xy
    half_width = max(half_width, abs(center_x - hotspot_x) + core_scale)
    half_height = max(half_height, abs(center_y - hotspot_y) + core_scale)

    x1 = max(context_x1, int(math.floor(center_x - half_width)))
    y1 = max(context_y1, int(math.floor(center_y - half_height)))
    x2 = min(context_x2, int(math.ceil(center_x + half_width)))
    y2 = min(context_y2, int(math.ceil(center_y + half_height)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return event.roi_xyxy
    return x1, y1, x2, y2


def _fixed_persistent_alarm_focus_roi(
    timeline: Sequence[CropFramePacket],
    event: AdaptiveEvent,
) -> Tuple[int, int, int, int]:


    context_x1, context_y1, context_x2, context_y2 = event.roi_xyxy
    boxes: List[Tuple[float, float, float, float]] = []
    focus_ids = event.focus_track_ids or [event.representative_track_id]
    for index in event.keyframe_indices:
        packet = timeline[index]
        current: List[Tuple[float, float, float, float]] = []
        for track_id in focus_ids:
            observation = packet.observations.get(track_id)
            if observation is not None:
                current.append(tuple(float(v) for v in observation["bbox_xyxy"]))
        if not current:
            for track_id in event.member_track_ids:
                observation = packet.observations.get(track_id)
                if observation is not None and _bbox_matches_region(
                    observation["bbox_xyxy"], event.core_roi_xyxy
                ):
                    current.append(tuple(float(v) for v in observation["bbox_xyxy"]))
        boxes.extend(current)


        nearest: Optional[Tuple[float, Tuple[float, float, float, float]]] = None
        for neighbor_id in event.neighbor_ids:
            observation = packet.observations.get(neighbor_id)
            if observation is None:
                continue
            proximity = 0.0
            for source_id in focus_ids:
                distance = packet.distance_map.get((source_id, neighbor_id), float("inf"))
                if math.isfinite(distance):
                    proximity = max(proximity, 1.0 / (1.0 + max(0.0, distance)))
            if proximity < 0.40:
                continue
            row = (float(proximity), tuple(float(v) for v in observation["bbox_xyxy"]))
            if nearest is None or row[0] > nearest[0]:
                nearest = row
        if nearest is not None:
            boxes.append(nearest[1])

    if not boxes:
        return event.roi_xyxy

    centers = np.asarray(
        [[0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])] for box in boxes],
        dtype=float,
    )
    scales = np.asarray([_bbox_center_scale(box)[2] for box in boxes], dtype=float)
    center_x = float(np.median(centers[:, 0]))
    center_y = float(np.median(centers[:, 1]))
    core_scale = max(20.0, float(event.core_scale), float(np.median(scales)))
    spread_x = (
        float(np.quantile(centers[:, 0], 0.85) - np.quantile(centers[:, 0], 0.15))
        if len(centers) > 1 else 0.0
    )
    spread_y = (
        float(np.quantile(centers[:, 1], 0.85) - np.quantile(centers[:, 1], 0.15))
        if len(centers) > 1 else 0.0
    )
    half_width = 0.85 * core_scale + 0.60 * math.sqrt(
        core_scale * max(core_scale, spread_x)
    )
    half_height = 0.85 * core_scale + 0.60 * math.sqrt(
        core_scale * max(core_scale, spread_y)
    )

    x1 = max(context_x1, int(math.floor(center_x - half_width)))
    y1 = max(context_y1, int(math.floor(center_y - half_height)))
    x2 = min(context_x2, int(math.ceil(center_x + half_width)))
    y2 = min(context_y2, int(math.ceil(center_y + half_height)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return event.roi_xyxy
    return x1, y1, x2, y2


def _frame_stop_focus_roi(
    packet: CropFramePacket,
    event: AdaptiveEvent,
) -> Tuple[int, int, int, int]:


    context_x1, context_y1, context_x2, context_y2 = event.roi_xyxy
    preferred = list(event.focus_track_ids) + list(event.member_track_ids)
    boxes = _hotspot_frame_boxes(
        packet,
        event.core_center_xy,
        event.core_scale,
        preferred,
        max_boxes=3,
    )
    if not boxes:
        return event.roi_xyxy
    scales = [_bbox_center_scale(box)[2] for box in boxes]
    median_scale = max(20.0, float(np.median(scales)))
    centers = np.asarray([_bbox_center_scale(box)[:2] for box in boxes], dtype=float)
    spread = max(
        float(np.ptp(centers[:, 0])) if len(centers) > 1 else 0.0,
        float(np.ptp(centers[:, 1])) if len(centers) > 1 else 0.0,
    )
    padding = math.sqrt(median_scale * median_scale + median_scale * max(median_scale, spread))
    min_x = min(box[0] for box in boxes) - padding
    min_y = min(box[1] for box in boxes) - padding
    max_x = max(box[2] for box in boxes) + padding
    max_y = max(box[3] for box in boxes) + padding

    x1 = max(context_x1, int(math.floor(min_x)))
    y1 = max(context_y1, int(math.floor(min_y)))
    x2 = min(context_x2, int(math.ceil(max_x)))
    y2 = min(context_y2, int(math.ceil(max_y)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return event.roi_xyxy
    return x1, y1, x2, y2


def _frame_focus_roi(
    packet: CropFramePacket,
    event: AdaptiveEvent,
) -> Tuple[int, int, int, int]:


    context_x1, context_y1, context_x2, context_y2 = event.roi_xyxy
    focus_ids = event.focus_track_ids or [event.representative_track_id]
    focus_boxes: List[Tuple[float, float, float, float]] = []
    for track_id in focus_ids:
        observation = packet.observations.get(track_id)
        if observation is not None:
            focus_boxes.append(tuple(float(v) for v in observation["bbox_xyxy"]))
    if not focus_boxes:
        for track_id in event.member_track_ids:
            observation = packet.observations.get(track_id)
            if observation is not None and _bbox_matches_region(
                observation["bbox_xyxy"],
                event.core_roi_xyxy,
            ):
                focus_boxes.append(tuple(float(v) for v in observation["bbox_xyxy"]))
    if not focus_boxes:
        return event.roi_xyxy

    focus_scales = [
        math.sqrt(max(1.0, (box[2] - box[0]) * (box[3] - box[1])))
        for box in focus_boxes
    ]
    median_scale = float(np.median(focus_scales))
    focus_centers = np.asarray(
        [[0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])] for box in focus_boxes],
        dtype=float,
    )
    center = np.median(focus_centers, axis=0)


    neighbor_candidates: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
    focus_set = set(focus_ids)
    for neighbor_id in event.neighbor_ids:
        if neighbor_id in focus_set:
            continue
        observation = packet.observations.get(neighbor_id)
        if observation is None:
            continue
        proximity = 0.0
        for source_id in focus_ids:
            distance = packet.distance_map.get((source_id, neighbor_id), float("inf"))
            if math.isfinite(distance):
                proximity = max(proximity, 1.0 / (1.0 + max(0.0, distance)))
        context_weight = float(event.neighbor_weights.get(neighbor_id, 0.0))


        relevance = math.sqrt(
            max(EPS, proximity * max(proximity, context_weight))
        )
        if relevance > 0.0:
            neighbor_candidates.append(
                (
                    int(neighbor_id),
                    float(relevance),
                    tuple(float(v) for v in observation["bbox_xyxy"]),
                )
            )
    neighbor_candidates.sort(key=lambda item: item[1], reverse=True)
    if neighbor_candidates:


        close_candidates = [item for item in neighbor_candidates if item[1] >= 0.40]
        context_boxes = []
        for _, relevance, box in close_candidates[:1]:
            bx1, by1, bx2, by2 = box
            neighbor_center = np.asarray(
                [0.5 * (bx1 + bx2), 0.5 * (by1 + by2)],
                dtype=float,
            )
            blended_center = center + 0.55 * clip01(relevance) * (neighbor_center - center)
            half_width = 0.30 * (bx2 - bx1) * clip01(relevance)
            half_height = 0.30 * (by2 - by1) * clip01(relevance)
            context_boxes.append(
                (
                    float(blended_center[0] - half_width),
                    float(blended_center[1] - half_height),
                    float(blended_center[0] + half_width),
                    float(blended_center[1] + half_height),
                )
            )
    else:
        context_boxes = []

    boxes = focus_boxes + context_boxes
    centers = np.asarray(
        [[0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])] for box in boxes],
        dtype=float,
    )
    spread = max(
        float(np.ptp(centers[:, 0])) if len(centers) > 1 else 0.0,
        float(np.ptp(centers[:, 1])) if len(centers) > 1 else 0.0,
    )
    padding = 0.80 * median_scale + 0.55 * math.sqrt(
        max(median_scale * max(0.0, spread), 0.0)
    )

    min_x = min(box[0] for box in boxes) - padding
    min_y = min(box[1] for box in boxes) - padding
    max_x = max(box[2] for box in boxes) + padding
    max_y = max(box[3] for box in boxes) + padding


    x1 = max(context_x1, int(math.floor(min_x)))
    y1 = max(context_y1, int(math.floor(min_y)))
    x2 = min(context_x2, int(math.ceil(max_x)))
    y2 = min(context_y2, int(math.ceil(max_y)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return event.roi_xyxy
    return x1, y1, x2, y2


def _packet_fps(timeline: Sequence[CropFramePacket]) -> float:
    if len(timeline) < 2:
        return 1.0
    first, last = timeline[0], timeline[-1]
    dt = max(EPS, last.time_sec - first.time_sec)
    return float(max(1.0, (last.frame_id - first.frame_id) / dt))


def _prepare_vlm_focus_image(image: np.ndarray) -> np.ndarray:


    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    short_side = min(height, width)
    long_side = max(height, width)
    if short_side < 384:
        scale = min(3.0, 384.0 / max(1.0, float(short_side)))
        if long_side * scale > 768:
            scale = 768.0 / max(1.0, float(long_side))
        image = cv2.resize(
            image,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_LANCZOS4,
        )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    return cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0.0)


def _outlined_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_labeled_box_in_crop(
    crop: np.ndarray,
    bbox_xyxy: Sequence[float],
    roi_xyxy: Sequence[int],
    color: Tuple[int, int, int],
    thickness: int,
    label: Optional[str] = None,
) -> None:


    rx1, ry1, _, _ = [int(v) for v in roi_xyxy]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1, x2 = x1 - rx1, x2 - rx1
    y1, y2 = y1 - ry1, y2 - ry1
    cv2.rectangle(crop, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not label:
        return
    origin = (x1, max(14, y1 - 4))
    cv2.putText(
        crop,
        str(label),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        crop,
        str(label),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_box_in_crop(
    crop: np.ndarray,
    bbox_xyxy: Sequence[float],
    roi_xyxy: Sequence[int],
    color: Tuple[int, int, int],
    thickness: int,
    label: Optional[str] = None,
) -> None:


    _draw_labeled_box_in_crop(
        crop,
        bbox_xyxy,
        roi_xyxy,
        color,
        thickness,
        label,
    )


def _draw_axis_in_crop(
    crop: np.ndarray,
    bbox_xyxy: Sequence[float],
    roi_xyxy: Sequence[int],
    result: AnomalyResult,
) -> None:
    rx1, ry1, _, _ = [int(v) for v in roi_xyxy]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    center_x = 0.5 * (x1 + x2) - rx1
    center_y = y2 - ry1
    scale = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
    length = float(np.clip(1.5 * scale, 18.0, 70.0))
    axis_x, axis_y = safe_unit(result.road_axis.x, result.road_axis.y)
    perp_x, perp_y = -axis_y, axis_x
    start = (int(round(center_x)), int(round(center_y)))
    cv2.arrowedLine(
        crop,
        start,
        (
            int(round(center_x + length * axis_x)),
            int(round(center_y + length * axis_y)),
        ),
        (255, 255, 0),
        2,
        cv2.LINE_AA,
        tipLength=0.22,
    )
    cv2.line(
        crop,
        (
            int(round(center_x - 0.65 * length * perp_x)),
            int(round(center_y - 0.65 * length * perp_y)),
        ),
        (
            int(round(center_x + 0.65 * length * perp_x)),
            int(round(center_y + 0.65 * length * perp_y)),
        ),
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def _select_vlm_comparison_positions(
    keyframes: Sequence[Dict[str, Any]],
    temporal_mode: str,
    images: Sequence[np.ndarray],
) -> List[int]:


    if not keyframes:
        return []
    role_to_positions: Dict[str, List[int]] = {}
    for position, row in enumerate(keyframes):
        role_to_positions.setdefault(str(row.get("role") or "transition"), []).append(position)

    chosen: List[int] = []


    if len(images) >= 2:
        change_scores: List[float] = []
        for left, right in zip(images[:-1], images[1:]):
            if left is None or right is None or left.size == 0 or right.size == 0:
                change_scores.append(0.0)
                continue
            a = cv2.cvtColor(cv2.resize(left, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(cv2.resize(right, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)

            delta = (b - float(np.median(b))) - (a - float(np.median(a)))
            change_scores.append(float(np.mean(np.abs(delta))))
        if change_scores:
            pivot = int(np.argmax(np.asarray(change_scores, dtype=float)))
            chosen.extend([pivot, pivot + 1])

    if temporal_mode == "stop_alarm_focus":
        priorities = ["pre_alarm", "first_alarm", "stop_peak", "post_peak", "end"]
    elif temporal_mode == "stop_cause_search":
        priorities = ["pre_stop", "first_stop_alarm", "stop_peak", "history_scan", "history_start"]
    elif temporal_mode == "persistent_alarm_focus":
        priorities = ["onset", "build_up_late", "peak", "post_alarm", "pre_alarm", "end"]
    elif temporal_mode == "forward_outcome_search":
        priorities = ["forward_alarm", "forward_early", "forward_mid", "forward_late", "forward_end"]
    else:
        priorities = ["interaction_before", "interaction", "consequence", "peak", "precursor", "end"]

    for role in priorities:
        candidates = role_to_positions.get(role, [])
        if not candidates:
            continue
        candidate = max(
            candidates,
            key=lambda index: float(keyframes[index].get("regional_score", 0.0)),
        )
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) >= 3:
            break

    if len(chosen) < 2:
        for candidate in (0, len(keyframes) - 1):
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) >= 2:
                break
    return sorted(chosen[:3])


def _compose_vlm_storyboard(
    images: Sequence[np.ndarray],
    max_side: int,
    force_single_row: bool = False,
) -> np.ndarray:


    separated: List[np.ndarray] = []
    for image in images:
        if image is None or image.size == 0:
            continue
        tile = cv2.copyMakeBorder(
            image,
            4,
            4,
            4,
            4,
            borderType=cv2.BORDER_CONSTANT,
            value=(235, 235, 235),
        )
        separated.append(tile)
    if not separated:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if force_single_row:
        target_width = min(max_side, 420 * len(separated))
        heights = [image.shape[0] for image in separated]
        common_height = max(160, min(420, int(np.median(heights))))
        resized = []
        for image in separated:
            scale = common_height / max(1, image.shape[0])
            resized.append(
                cv2.resize(
                    image,
                    (max(1, int(round(image.shape[1] * scale))), common_height),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
            )
        canvas = np.concatenate(resized, axis=1)
        if canvas.shape[1] > target_width:
            scale = target_width / canvas.shape[1]
            canvas = cv2.resize(
                canvas,
                (target_width, max(1, int(round(canvas.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return canvas
    return _compose_image_grid(
        separated,
        max_side=max_side,
        title=None,
        background=(235, 235, 235),
    )


def _compose_image_grid(
    images: Sequence[np.ndarray],
    max_side: int,
    title: Optional[str],
    background: Tuple[int, int, int],
) -> np.ndarray:
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    count = len(images)
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    aspect_values = [image.shape[1] / max(image.shape[0], 1) for image in images]
    aspect = float(np.median(aspect_values)) if aspect_values else 1.5
    cell_width = max(160, int(max_side / max(columns, 1)))
    cell_height = max(100, int(round(cell_width / max(aspect, 0.25))))
    title_height = 42 if title else 0
    canvas = np.full(
        (title_height + rows * cell_height, columns * cell_width, 3),
        background,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        scale = min(
            cell_width / max(image.shape[1], 1),
            cell_height / max(image.shape[0], 1),
        )
        new_width = max(1, int(round(image.shape[1] * scale)))
        new_height = max(1, int(round(image.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        x = column * cell_width + (cell_width - new_width) // 2
        y = title_height + row * cell_height + (cell_height - new_height) // 2
        canvas[y : y + new_height, x : x + new_width] = resized
    if title:
        cv2.putText(
            canvas,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    longest = max(canvas.shape[:2])
    if longest > max_side:
        scale = max_side / longest
        canvas = cv2.resize(
            canvas,
            (
                max(1, int(round(canvas.shape[1] * scale))),
                max(1, int(round(canvas.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return canvas
