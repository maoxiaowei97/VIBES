from __future__ import annotations


import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
from tqdm import tqdm

from .anomaly import compute_motion_anomaly
from .common import EPS, AnomalyResult, TrackState, VideoRunResult
from .config import ensure_dir
from .crop_cache import CropAnalysisCacheWriter
from .crop_stage import run_crop_stage
from .detection import detect_frame
from .road import build_road_direction_field
from .tracking import (
    assign_track_ids,
    build_byte_tracker,
    build_neighbor_map,
    motion_snapshot,
    update_track_states,
)
from .visualization import draw_overlay, scale_rows


LEGACY_ANALYSIS_FRAME_STRIDE = 3


@dataclass
class PerformanceStats:
    processed_frames: int = 0
    rescue_frames: int = 0
    stage_seconds: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, name: str, seconds: float) -> None:
        self.stage_seconds[name] += float(seconds)

    def print_summary(
        self,
        video_name: str,
        source_frames: int,
        wall_seconds: float,
        fps: float,
    ) -> None:
        source_rate = source_frames / max(wall_seconds, EPS)
        print(
            f"[PERF] {video_name}: wall={wall_seconds:.2f}s, "
            f"source_fps={source_rate:.2f}, realtime={source_rate / max(fps, EPS):.2f}x, "
            f"analysis_frames={self.processed_frames}, far_rescue={self.rescue_frames}"
        )
        for stage in ("decode", "detect", "track", "anomaly", "csv", "cache", "render"):
            seconds = float(self.stage_seconds.get(stage, 0.0))
            milliseconds = 1000.0 * seconds / max(1, self.processed_frames)
            print(
                f"[PERF]   {stage:>8s}: {seconds:7.2f}s | "
                f"{milliseconds:7.2f} ms/analysis-frame"
            )


def _csv_fieldnames() -> List[str]:
    return [
        "frame_id",
        "time_sec",
        "track_id",
        "event",
        "score",
        "confidence",
        "lateral_score",
        "overspeed_score",
        "decel_score",
        "stop_score",
        "v_parallel",
        "v_perpendicular",
        "road_source",
        "road_confidence",
        "context_count",
        "det_conf",
        "overall_confidence",
        "detection_confidence",
        "track_confidence",
        "axis_confidence",
        "motion_consistency",
        "context_confidence",
        "stop_context_confidence",
        "stop_context_count",
        "current_flow_speed",
        "current_flow_noise",
        "current_flow_confidence",
        "historical_flow_speed",
        "historical_flow_noise",
        "historical_flow_confidence",
        "historical_geometry_confidence",
        "historical_flow_support",
        "effective_flow_corridor_membership",
        "current_flow_road_membership",
        "cold_start_flow_road_membership",
        "robust_track_speed",
        "robust_track_noise",
        "robust_cluster_stability",
        "robust_duration_reliability",
        "robust_motion_identity",
        "stationary_peer_mass",
        "stationary_peer_count",
        "cold_start_isolation_gate",
        "current_vehicle_identity",
        "historical_vehicle_identity",
        "road_vehicle_identity",
        "vehicle_candidate_confidence",
        "static_nonvehicle_suspect",
        "display_eligible",
        "mu_parallel",
        "sigma_parallel",
        "mu_perpendicular",
        "sigma_perpendicular",
        "motion_resolution",
        "motion_innovation",
        "recent_window",
        "z_lateral_speed",
        "z_lateral_travel",
        "effective_lateral_z",
        "lateral_angle_deg",
        "lateral_fraction",
        "lateral_travel",
        "lateral_sign_coherence",
        "forward_retention",
        "forward_preservation",
        "lateral_endpoint",
        "lateral_persistence",
        "lateral_recovery",
        "lateral_early",
        "lateral_decay_recovery",
        "side_slip_likelihood",
        "lane_change_likelihood",
        "normal_lane_change_alternative",
        "lateral_raw_score",
        "instantaneous_lateral_score",
        "persistent_lateral_score",
        "lateral_hazard_gate",
        "lateral_memory_input_z",
        "temporal_lateral_z",
        "temporal_lateral_confidence",
        "temporal_lateral_coherence",
        "lateral_memory_effective_count",
        "lateral_memory_count",
        "lateral_memory_window",
        "lane_change_recovery_evidence",
        "lateral_history_retention",
        "lateral_confidence",
        "overspeed_z",
        "overspeed_ratio",
        "overspeed_confidence",
        "old_parallel",
        "new_parallel",
        "decel_z",
        "drop_ratio",
        "decel_confidence",
        "recent_speed",
        "target_speed_noise",
        "staticness",
        "stop_displacement",
        "position_stability",
        "flow_speed",
        "traffic_occupancy_confidence",
        "flow_corridor_membership",
        "flow_road_membership",
        "stop_z",
        "flow_stop_z",
        "flow_cumulative_z",
        "flow_context_effective_count",
        "flow_context_support_reliability",
        "flow_context_confidence",
        "flow_low_speed_persistence",
        "stop_state_gate",
        "flow_road_gate",
        "transition_stop_z",
        "cumulative_stop_z",
        "flow_stop_score",
        "transition_stop_score",
        "persistent_stop_score",
        "persistent_flow_stop_score",
        "persistent_flow_stop_raw_score",
        "persistent_flow_stop_z",
        "persistent_flow_stop_confidence",
        "persistent_flow_stop_effective_count",
        "persistent_flow_stop_count",
        "persistent_flow_stop_window",
        "cold_start_stop_score",
        "cold_start_stop_raw_score",
        "cold_start_stop_z",
        "cold_start_cumulative_z",
        "cold_start_stop_confidence",
        "cold_start_stationary_gate",
        "cold_reference_speed",
        "cold_reference_noise",
        "cold_reference_confidence",
        "stable_near_stop",
        "flow_stop_effective_z",
        "transition_stop_effective_z",
        "temporal_stop_z",
        "temporal_stop_confidence",
        "stop_memory_effective_count",
        "stop_memory_count",
        "stop_memory_window",
        "moving_reference",
        "moving_reference_noise",
        "moving_reference_confidence",
        "prior_motion_evidence",
        "low_speed_persistence",
        "stop_confidence",
        "strongest_longitudinal_event",
        "strongest_longitudinal_score",
        "collision_lateral_support",
        "collision_joint_energy",
        "collision_balance",
        "collision_gate",
        "collision_joint_score",
        "impact_continuation_support",
        "impact_retention",
        "impact_posterior_score",
        "impact_episode_score",
    ]


def _csv_row(
    frame_id: int,
    fps: float,
    track_id: int,
    detection_confidence: float,
    result: AnomalyResult,
) -> Dict[str, Any]:
    debug = result.debug
    row: Dict[str, Any] = {
        "frame_id": frame_id,
        "time_sec": f"{frame_id / fps:.3f}",
        "track_id": track_id,
        "event": result.event,
        "score": f"{result.score:.6f}",
        "confidence": f"{result.confidence:.4f}",
        "lateral_score": f"{result.lateral_score:.6f}",
        "overspeed_score": f"{result.overspeed_score:.6f}",
        "decel_score": f"{result.decel_score:.6f}",
        "stop_score": f"{result.stop_score:.6f}",
        "v_parallel": f"{result.v_parallel:.7f}",
        "v_perpendicular": f"{result.v_perpendicular:.7f}",
        "road_source": result.road_axis.source,
        "road_confidence": f"{result.road_axis.confidence:.4f}",
        "context_count": result.context_count,
        "det_conf": f"{detection_confidence:.4f}",
    }
    for key in _csv_fieldnames():
        if key in row:
            continue
        value = debug.get(key, 0.0)
        if isinstance(value, int):
            row[key] = value
        elif isinstance(value, float):
            row[key] = f"{value:.7f}"
        else:
            row[key] = value
    return row


def run_video(
    video_path: Path,
    output_dir: Path,
    model: Any,
    class_ids_keep: set[int],
    args: argparse.Namespace,
) -> VideoRunResult:
    ensure_dir(output_dir)
    road_field = build_road_direction_field(video_path, output_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = max(1.0, float(capture.get(cv2.CAP_PROP_FPS)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total_frames = min(total_frames, int(args.max_frames))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(args.frame_stride))
    sampling_interval_scale = max(
        1.0, float(stride) / float(LEGACY_ANALYSIS_FRAME_STRIDE)
    )


    effective_history_size = max(
        6,
        int(math.ceil(float(args.history_size) / sampling_interval_scale)),
    )
    tracker = build_byte_tracker(fps / stride)
    states: Dict[int, TrackState] = {}
    print(
        f"[INFO] Temporal sampling: source_fps={fps:.2f}, stride={stride}, "
        f"analysis_fps={fps / stride:.2f}, rate_vs_legacy={1.0 / sampling_interval_scale:.2f}x, "
        f"history={effective_history_size} obs (legacy-equivalent {int(args.history_size)})"
    )

    render_scale = 1.0
    if args.output_max_side > 0 and max(frame_width, frame_height) > args.output_max_side:
        render_scale = float(args.output_max_side) / max(frame_width, frame_height)
    render_width = max(2, int(round(frame_width * render_scale)))
    render_height = max(2, int(round(frame_height * render_scale)))
    video_output = output_dir / "annotated_video.mp4"
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / stride),
        (render_width, render_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频: {video_output}")

    csv_path = output_dir / "motion_scores.csv"
    csv_handle = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_handle, fieldnames=_csv_fieldnames())
    csv_writer.writeheader()

    crop_enabled = not bool(args.disable_adaptive_crops)
    vlm_enabled = bool(args.enable_vlm) and crop_enabled
    if bool(args.enable_vlm) and not crop_enabled:
        print("[WARN] 本地 VLM 依赖 adaptive crops；当前已关闭 VLM。")
    elif vlm_enabled:
        print(
            "[INFO] 严格三阶段执行：检测/跟踪/惊奇 -> 完整生成 Crop -> "
            "逐 event 原始 Qwen3-VL 推理。"
        )

    cache_path = output_dir / ".crop_analysis_cache.pkl"
    cache_writer = (
        CropAnalysisCacheWriter(
            cache_path,
            video_path=video_path,
            fps=fps,
            frame_stride=stride,
        )
        if crop_enabled
        else None
    )

    frame_id = -1
    analysis_step = -1
    peak_score = 0.0
    peak_frame = 0
    event_frames: List[int] = []
    performance = PerformanceStats()
    wall_start = time.perf_counter()

    progress = tqdm(
        total=total_frames,
        desc=f"{video_path.name} v12-cold-start-stop",
        unit="frame",
        dynamic_ncols=True,
    )
    try:
        while frame_id + 1 < total_frames:
            start = time.perf_counter()
            ok = capture.grab()
            if not ok:
                break
            frame_id += 1
            progress.update(1)
            if frame_id % stride != 0:
                performance.add("decode", time.perf_counter() - start)
                continue

            ok, frame = capture.retrieve()
            performance.add("decode", time.perf_counter() - start)
            if not ok or frame is None:
                break

            analysis_step += 1
            performance.processed_frames += 1
            run_rescue = bool(
                args.far_rescue_interval > 0
                and analysis_step % max(1, int(args.far_rescue_interval)) == 0
            )

            start = time.perf_counter()
            detections = detect_frame(
                frame,
                model,
                class_ids_keep,
                global_max_side=int(args.global_max_side),
                run_far_rescue=run_rescue,
                far_slice_size=int(args.far_slice_size),
            )
            performance.add("detect", time.perf_counter() - start)
            if run_rescue:
                performance.rescue_frames += 1

            start = time.perf_counter()
            observations = assign_track_ids(detections, tracker, frame.shape[:2])
            performance.add("track", time.perf_counter() - start)

            update_track_states(states, observations, frame_id, effective_history_size)
            active_ids = {int(row["track_id"]) for row in observations}
            stale_gap = max(30, int(round(2.5 * fps)))
            for track_id in list(states):
                if (
                    track_id not in active_ids
                    and frame_id - states[track_id].last_frame > stale_gap
                ):
                    del states[track_id]

            motions = {
                track_id: motion_snapshot(states[track_id])
                for track_id in active_ids
                if track_id in states
            }


            road_field.update_traffic_occupancy(
                states,
                motions,
                sorted(active_ids),
                sampling_interval_scale=sampling_interval_scale,
            )

            neighbor_map, distance_map = build_neighbor_map(observations)

            start = time.perf_counter()
            results: Dict[int, AnomalyResult] = {}
            frame_csv_rows: List[Dict[str, Any]] = []
            for row in observations:
                track_id = int(row["track_id"])
                result = compute_motion_anomaly(
                    track_id,
                    states,
                    motions,
                    neighbor_map.get(track_id, []),
                    distance_map,
                    road_field,
                    sampling_interval_scale=sampling_interval_scale,
                )
                results[track_id] = result

                if result.score > peak_score:
                    peak_score = float(result.score)
                    peak_frame = int(frame_id)
                if result.score >= args.anomaly_threshold:
                    event_frames.append(frame_id)
                frame_csv_rows.append(
                    _csv_row(
                        frame_id,
                        fps,
                        track_id,
                        float(row["det_conf"]),
                        result,
                    )
                )
            performance.add("anomaly", time.perf_counter() - start)


            start = time.perf_counter()
            if frame_csv_rows:
                csv_writer.writerows(frame_csv_rows)
            performance.add("csv", time.perf_counter() - start)


            start = time.perf_counter()
            if cache_writer is not None:
                cache_writer.add(
                    frame_id=frame_id,
                    observations=observations,
                    results=results,
                    neighbor_map=neighbor_map,
                    distance_map=distance_map,
                )
            performance.add("cache", time.perf_counter() - start)

            start = time.perf_counter()
            if render_scale < 0.999:
                render_frame = cv2.resize(
                    frame,
                    (render_width, render_height),
                    interpolation=cv2.INTER_AREA,
                )
                render_observations = scale_rows(observations, render_scale)
            else:
                render_frame = frame
                render_observations = observations
            overlay = draw_overlay(
                render_frame,
                render_observations,
                results,
                threshold=float(args.anomaly_threshold),
                show_axes=not bool(args.hide_road_axes),
            )
            writer.write(overlay)
            performance.add("render", time.perf_counter() - start)
    finally:
        progress.close()
        if cache_writer is not None:
            cache_writer.close()
        csv_handle.close()
        writer.release()
        capture.release()


    road_field.save_traffic_debug(output_dir)

    wall_seconds = time.perf_counter() - wall_start
    performance.print_summary(
        video_path.name,
        max(0, frame_id + 1),
        wall_seconds,
        fps,
    )

    if cache_writer is not None:
        run_crop_stage(
            video_path=video_path,
            output_dir=output_dir,
            cache_path=cache_path,
            record_count=cache_writer.record_count,
            fps=fps,
            args=args,
        )

    cooldown = max(1, int(round(fps)))
    merged_events: List[int] = []
    for event_frame in sorted(set(event_frames)):
        if not merged_events or event_frame - merged_events[-1] > cooldown:
            merged_events.append(event_frame)

    print(f"[INFO] Annotated video: {video_output}")
    print(f"[INFO] Motion scores: {csv_path}")
    if not bool(args.disable_adaptive_crops):
        print(f"[INFO] Adaptive crops: {output_dir / 'adaptive_spatiotemporal_crops'}")
    if vlm_enabled:
        print(
            "[INFO] VLM 将在全部 Crop 完成后逐 event 运行；结果文件名包含 "
            "_vlm_raw_prompt_test_<prompt_hash>.json。"
        )
    return VideoRunResult(
        video_path=video_path,
        peak_frame=peak_frame,
        peak_score=peak_score,
        event_count=len(merged_events),
        output_dir=output_dir,
    )
