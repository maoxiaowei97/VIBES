from __future__ import annotations


import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import torch

from .common import DEFAULT_CLASS_IDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_vlm_device() -> str:
    if torch.cuda.is_available():


        return "cuda:0"
    return "cpu"


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="ZeroTAD v12: detection/surprise -> crop -> per-event raw Qwen3-VL."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="单个 mp4，或包含 mp4 的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / timestamp,
        help="输出根目录；为避免覆盖，要求运行前不存在。",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "yolo26l.engine",
        help="Ultralytics .pt 或 TensorRT .engine 模型。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="目标检测设备，例如 cuda:0。",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="全图检测置信度阈值。")
    parser.add_argument("--class-ids", type=str, default=DEFAULT_CLASS_IDS)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="分析帧步长；速度使用真实 frame_id 间隔归一化。",
    )
    parser.add_argument("--global-max-side", type=int, default=1280)
    parser.add_argument(
        "--far-rescue-interval",
        type=int,
        default=2,
        help="远端 SAHI 补检间隔，单位为 analysis frame。",
    )
    parser.add_argument("--far-slice-size", type=int, default=640)
    parser.add_argument(
        "--history-size",
        type=int,
        default=12,
        help="等效轨迹历史预算；运行时会随 frame-stride 换算实际保存观测数。",
    )
    parser.add_argument("--anomaly-threshold", type=float, default=1.25)
    parser.add_argument("--output-max-side", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--hide-road-axes", action="store_true")
    parser.add_argument(
        "--crop-interval-seconds",
        type=float,
        default=8.0,
        help="事件组织 Segment 时长（秒）。默认 8 秒；每个 Segment 最多保留 4 个 Event。",
    )
    parser.add_argument(
        "--focus-base-seconds",
        type=float,
        default=4.0,
        help=(
            "自适应 Crop 的基础时间尺度。默认仍为旧版 4 秒，只控制前兆/后果窗口，"
            "与 8 秒 Segment 解耦，避免 Segment 变长后把 Storyboard 时间范围也翻倍。"
        ),
    )
    parser.add_argument(
        "--max-events-per-segment",
        type=int,
        default=4,
        help="每个 8 秒 Segment 最多保留现有优先级排序中的前 4 个 Crop event。",
    )
    parser.add_argument("--disable-adaptive-crops", action="store_true")

    vlm_toggle = parser.add_mutually_exclusive_group()
    vlm_toggle.add_argument("--enable-vlm", dest="enable_vlm", action="store_true")
    vlm_toggle.add_argument("--disable-vlm", dest="enable_vlm", action="store_false")
    parser.set_defaults(enable_vlm=True)
    parser.add_argument(
        "--vlm-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct",
        help="本地 Qwen3-VL-8B-Instruct 目录。",
    )
    parser.add_argument(
        "--vlm-device",
        type=str,
        default=_default_vlm_device(),
        help="本地 VLM 设备。默认 cuda:0；检测器会在 VLM 加载前释放。",
    )
    parser.add_argument(
        "--vlm-image-max-side",
        type=int,
        default=896,
        help="Qwen3-VL 每张故事板/对比图的视觉像素预算基准；与独立验证脚本保持一致。",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=96,
        help=(
            "每个 event 原始自然语言回答的最大生成 token。默认 96；配合 1～2 句话 Prompt，"
            "减少逐格复述，同时保留碰撞、侧滑、道路中停止和冒烟判断。"
        ),
    )
    parser.add_argument(
        "--vlm-attention",
        choices=("auto", "flash_attention_2", "sdpa"),
        default="sdpa",
        help="Qwen3-VL attention 实现。默认 sdpa 更利于重复运行的一致性；追求极限速度可改 flash_attention_2。",
    )
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=6,
        choices=tuple(range(1, 9)),
        help=(
            "VLM micro-batch。RTX 5090/32GB 默认 6，目标是把吞吐压到约 0.9～1.0 秒/event；"
            "显存不足时优先回退到 4，再继续拆分。设为 1 可做严格逐 event 对照。"
        ),
    )
    parser.add_argument(
        "--vlm-prefetch-depth",
        type=int,
        default=2,
        help=(
            "后台 CPU 预处理队列深度。默认 2；batch>1 时预处理下一批并与当前 GPU 生成重叠，"
            "batch=1 时则预取下一个 event。"
        ),
    )
    parser.add_argument(
        "--vlm-execution-mode",
        choices=("deferred", "online"),
        default="deferred",
        help=(
            "兼容旧命令保留；当前无论取值如何，都严格在全部 Crop 完成并释放检测器后再运行 Qwen。"
        ),
    )
    parser.add_argument("--vlm-overwrite", action="store_true")

    parser.add_argument(
        "--vlm-prompt-file",
        type=Path,
        default=None,
        help="可选 UTF-8 prompt 文件；未指定时使用代码内置 prompt。",
    )
    parser.add_argument(
        "--vlm-output-tag",
        type=str,
        default="raw_prompt_test",
        help="VLM 结果文件标签；文件名还会自动加入 prompt 哈希。",
    )
    parser.add_argument(
        "--vlm-allow-storyboard-only",
        action="store_true",
        help="comparison 缺失时允许只用 storyboard 推理；默认跳过该 event。",
    )
    print_group = parser.add_mutually_exclusive_group()
    print_group.add_argument(
        "--vlm-print-answers",
        dest="vlm_print_answers",
        action="store_true",
        help="在终端逐 event 打印完整原始回答；会增加 SSH/终端输出开销。",
    )
    print_group.add_argument(
        "--vlm-no-print-answers",
        dest="vlm_print_answers",
        action="store_false",
        help="不在终端打印完整回答；结果仍完整写入 JSON/CSV。",
    )
    parser.set_defaults(vlm_print_answers=False)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_class_ids(text: str) -> set[int]:
    values = {int(token.strip()) for token in text.split(",") if token.strip()}
    if not values:
        raise ValueError("--class-ids 不能为空。")
    return values


def discover_videos(path: Path, max_videos: int) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    if path.is_file():
        return [path]
    videos = sorted(path.glob("*.mp4"))
    if max_videos > 0:
        videos = videos[:max_videos]
    if not videos:
        raise FileNotFoundError(f"目录下没有 mp4: {path}")
    return videos
