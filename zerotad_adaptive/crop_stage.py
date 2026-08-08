from __future__ import annotations


import argparse
import os
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from .adaptive_crop import AdaptiveCropWriter
from .common import EPS
from .crop_cache import iter_crop_records


def run_crop_stage(
    video_path: Path,
    output_dir: Path,
    cache_path: Path,
    record_count: int,
    fps: float,
    args: argparse.Namespace,
) -> float:


    crop_writer = AdaptiveCropWriter(
        output_dir=output_dir,
        fps=fps,
        interval_seconds=float(args.crop_interval_seconds),
        anomaly_threshold=float(args.anomaly_threshold),
        canvas_max_side=int(args.output_max_side),
        max_events_per_segment=int(args.max_events_per_segment),
        focus_base_seconds=float(getattr(args, "focus_base_seconds", 4.0)),
        enabled=True,
        on_segment_ready=None,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        crop_writer.close()
        raise RuntimeError(f"Crop 阶段无法重新打开视频: {video_path}")

    started = time.perf_counter()
    cursor = -1
    progress = tqdm(
        total=max(0, int(record_count)),
        desc=f"{video_path.name} adaptive-crop",
        unit="analysis-frame",
        dynamic_ncols=True,
    )
    success = False
    try:
        for record in iter_crop_records(cache_path):
            while cursor < int(record.frame_id):
                if not capture.grab():
                    raise RuntimeError(
                        f"Crop 阶段读取视频提前结束: frame={record.frame_id}"
                    )
                cursor += 1
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"Crop 阶段解码失败: frame={record.frame_id}")
            crop_writer.add_frame(
                frame=frame,
                frame_id=record.frame_id,
                observations=record.observations,
                results=record.results,
                neighbor_map=record.neighbor_map,
                distance_map=record.distance_map,
            )
            progress.update(1)


        crop_writer.close()
        success = True
    except BaseException:
        crop_writer.abort()
        raise
    finally:
        progress.close()
        capture.release()

    elapsed = time.perf_counter() - started
    if success:
        try:
            os.remove(cache_path)
        except OSError:
            pass
    print(
        f"[PERF] {video_path.name}: adaptive-crop={elapsed:.2f}s, "
        f"frames={record_count}, rate={record_count / max(elapsed, EPS):.2f} analysis-fps"
    )
    return elapsed
