from __future__ import annotations

import csv
import gc
import time

import torch

from .common import VideoRunResult
from .config import discover_videos, ensure_dir, parse_args, parse_class_ids
from .detection import build_detection_model
from .runner import run_video
from .vlm_local_inference import diagnose_existing_output


def _release_detector(model: object) -> None:
    try:
        if hasattr(model, "predictor"):
            model.predictor = None
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请使用新目录: {args.output_dir}")

    videos = discover_videos(args.input_path, int(args.max_videos))
    ensure_dir(args.output_dir)
    class_ids_keep = parse_class_ids(args.class_ids)
    model = build_detection_model(args.detector_model, float(args.conf), str(args.device))

    results: list[VideoRunResult] = []
    for video_path in videos:
        video_output = args.output_dir / video_path.stem
        print(f"\n[INFO] Processing: {video_path}")
        result = run_video(video_path, video_output, model, class_ids_keep, args)
        results.append(result)
        print(
            f"[INFO] {video_path.stem}: peak_frame={result.peak_frame}, "
            f"peak_score={result.peak_score:.4f}, events={result.event_count}"
        )


    if bool(args.enable_vlm) and not bool(args.disable_adaptive_crops):
        print("\n[VLM-RAW] 全部检测和 Crop 已完成，正在释放检测器资源……")
        _release_detector(model)
        model = None
        print(
            f"[VLM-RAW] 开始 Crop 后原始推理: model={args.vlm_model_path}, "
            f"device={args.vlm_device}, prompt={args.vlm_prompt_file}"
        )
        started = time.perf_counter()
        try:


            summary = diagnose_existing_output(
                video_output_dir=args.output_dir,
                model_path=args.vlm_model_path,
                device=str(args.vlm_device),
                image_max_side=int(args.vlm_image_max_side),
                max_new_tokens=int(args.vlm_max_new_tokens),
                attention=str(args.vlm_attention),
                overwrite=bool(args.vlm_overwrite),
                batch_size=int(args.vlm_batch_size),
                max_events_per_segment=0,
                prompt_file=args.vlm_prompt_file,
                tag=str(args.vlm_output_tag),
                allow_storyboard_only=bool(args.vlm_allow_storyboard_only),
                print_answers=bool(args.vlm_print_answers),
                prefetch_depth=int(args.vlm_prefetch_depth),
            )
            print(
                f"[VLM-RAW] all cases: "
                f"discovered={summary.get('discovered_event_count', 0)}, "
                f"new={summary.get('new_inference_count', 0)}, "
                f"cached={summary.get('skipped_cached_count', 0)}, "
                f"time={time.perf_counter() - started:.2f}s"
            )
        except Exception as exc:
            print(
                "[WARN] 逐 event VLM 失败；检测、异常结果与全部 Crop 已完整保存: "
                f"{exc}"
            )
    else:
        _release_detector(model)
        model = None

    summary_path = args.output_dir / "batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_name",
                "video_path",
                "peak_frame",
                "peak_score",
                "event_count",
                "output_dir",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_name": result.video_path.stem,
                    "video_path": str(result.video_path),
                    "peak_frame": result.peak_frame,
                    "peak_score": f"{result.peak_score:.6f}",
                    "event_count": result.event_count,
                    "output_dir": str(result.output_dir),
                }
            )
    print(f"[INFO] Batch summary: {summary_path}")
    return 0
