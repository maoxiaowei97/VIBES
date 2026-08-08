from __future__ import annotations


import csv
import hashlib
import importlib.util
import json
import re
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


DEFAULT_PROMPT = (
    "这两张图来自同一段交通视频。第一张是关键时刻对比图，第二张是按时间排列的故事板。"
    "每个格子都是不同时间帧，时间顺序为从左到右、从上到下。"
    "请用1至2句话，先说明主要车辆发生了什么，再给出判断；不要逐格复述，也不要罗列无关车辆。"
    "若看到碰撞、侧滑或横向失控、车辆停止在道路中、或者冒烟等情况，请明确说出；"
    "若没有看到明确事件，直接说明，注意有些是正常跟车，超车，变道，保证真正异常情况报出的前提下，降低误报。不要输出JSON。"
)

_MIN_PIXELS = 256 * 28 * 28
_STORYBOARD_SUFFIX = "_vlm_storyboard.jpg"
_COMPARISON_SUFFIX = "_vlm_comparison.jpg"


@dataclass(frozen=True)
class EventPair:
    case_name: str
    segment_name: str
    event_name: str
    event_prefix: Path
    comparison_path: Optional[Path]
    storyboard_path: Path
    metadata_path: Optional[Path]

    @property
    def input_paths(self) -> Tuple[Path, ...]:
        paths: List[Path] = []
        if self.comparison_path is not None:
            paths.append(self.comparison_path)
        paths.append(self.storyboard_path)
        return tuple(paths)


@dataclass(frozen=True)
class PreparedEvent:
    event: EventPair
    inputs_cpu: Any
    preprocess_seconds: float
    processor_path: str
    input_tokens: int = 0


@dataclass(frozen=True)
class PreparedBatch:
    events: Tuple[EventPair, ...]
    inputs_cpu: Any
    preprocess_seconds: float
    processor_path: str
    input_tokens: Tuple[int, ...] = ()


@dataclass(frozen=True)
class RawResult:
    event: EventPair
    raw_text: str
    preprocess_seconds: float
    generate_seconds: float
    decode_seconds: float
    wall_seconds: float
    transfer_seconds: float = 0.0
    pipeline_wait_seconds: float = 0.0
    batch_size: int = 1
    batch_wall_seconds: float = 0.0
    timing_mode: str = "per_event_exact"
    processor_path: str = "standalone_process_vision_info"
    input_tokens: int = 0
    output_tokens: int = 0
    hit_max_new_tokens: bool = False
    batch_preprocess_seconds: float = 0.0
    batch_transfer_seconds: float = 0.0
    batch_generate_seconds: float = 0.0
    batch_decode_seconds: float = 0.0
    batch_total_output_tokens: int = 0
    batch_max_output_tokens: int = 0

    @property
    def total_seconds(self) -> float:


        return (
            self.preprocess_seconds
            + self.transfer_seconds
            + self.generate_seconds
            + self.decode_seconds
        )

    @property
    def pipeline_blocking_seconds(self) -> float:
        if self.batch_size > 1:
            return self.total_seconds
        return (
            self.pipeline_wait_seconds
            + self.transfer_seconds
            + self.generate_seconds
            + self.decode_seconds
        )

    @property
    def batch_output_tokens_per_second(self) -> float:
        if self.batch_generate_seconds <= 0:
            return 0.0
        return self.batch_total_output_tokens / self.batch_generate_seconds


def load_prompt(prompt_file: Optional[Path]) -> str:
    if prompt_file is None:
        prompt = DEFAULT_PROMPT
    else:
        path = Path(prompt_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prompt 文件不存在: {path}")
        prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    return prompt


def safe_tag(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text).strip())
    return value.strip("._") or "raw_prompt_test"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]


def choose_attention(device_text: str, requested: str) -> str:
    if requested != "auto":
        return requested
    use_cuda = device_text.startswith("cuda") and torch.cuda.is_available()
    if use_cuda and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def sanitize_generation_config(model: Any) -> None:
    config = model.generation_config
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.num_beams = 1
    config.use_cache = True


_sanitize_model_generation_config = sanitize_generation_config


def find_crop_roots(root: Path) -> List[Path]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root}")

    if root.name == "adaptive_spatiotemporal_crops" and root.is_dir():
        return [root]

    direct = root / "adaptive_spatiotemporal_crops"
    if direct.is_dir():
        return [direct.resolve()]

    crop_roots = sorted(
        path.resolve()
        for path in root.glob("*/adaptive_spatiotemporal_crops")
        if path.is_dir()
    )
    if crop_roots:
        return crop_roots

    return sorted(
        path.resolve()
        for path in root.glob("*/*/adaptive_spatiotemporal_crops")
        if path.is_dir()
    )


def discover_events(
    run_root: Path,
    selected_cases: Sequence[str] = (),
    selected_segments: Sequence[str] = (),
    event_regex: str = "",
    allow_storyboard_only: bool = False,
) -> Tuple[List[EventPair], List[str]]:
    case_filter = {item.strip() for item in selected_cases if item.strip()}
    segment_filter = {item.strip() for item in selected_segments if item.strip()}
    pattern = re.compile(event_regex) if event_regex else None

    events: List[EventPair] = []
    warnings: List[str] = []
    crop_roots = find_crop_roots(run_root)
    if not crop_roots:
        raise FileNotFoundError(
            f"未找到 adaptive_spatiotemporal_crops: {Path(run_root).expanduser().resolve()}"
        )

    for crop_root in crop_roots:
        case_name = crop_root.parent.name
        if case_filter and case_name not in case_filter:
            continue

        for segment_dir in sorted(path for path in crop_root.glob("segment_*") if path.is_dir()):
            if segment_filter and segment_dir.name not in segment_filter:
                continue

            for storyboard in sorted(segment_dir.glob(f"event_*{_STORYBOARD_SUFFIX}")):
                event_name = storyboard.name[: -len(_STORYBOARD_SUFFIX)]
                if pattern is not None and pattern.search(event_name) is None:
                    continue

                prefix = storyboard.with_name(event_name)
                comparison = storyboard.with_name(event_name + _COMPARISON_SUFFIX)
                if comparison.is_file():
                    comparison_path: Optional[Path] = comparison.resolve()
                elif allow_storyboard_only:
                    comparison_path = None
                    warnings.append(f"comparison 缺失，仅使用 storyboard: {storyboard}")
                else:
                    warnings.append(f"跳过 comparison 缺失事件: {storyboard}")
                    continue

                metadata = storyboard.with_name(event_name + ".json")
                events.append(
                    EventPair(
                        case_name=case_name,
                        segment_name=segment_dir.name,
                        event_name=event_name,
                        event_prefix=prefix.resolve(),
                        comparison_path=comparison_path,
                        storyboard_path=storyboard.resolve(),
                        metadata_path=metadata.resolve() if metadata.is_file() else None,
                    )
                )

    events.sort(key=lambda item: (item.case_name, item.segment_name, item.event_name))
    return events, warnings


def _actual_input_tokens(inputs: Any) -> List[int]:
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if attention_mask is None:
        input_ids = inputs["input_ids"]
        return [int(input_ids.shape[-1])] * int(input_ids.shape[0])
    return [int(value) for value in attention_mask.sum(dim=-1).tolist()]


def _actual_generated_tokens(
    generated_ids: Sequence[torch.Tensor],
    pad_token_id: Optional[int],
    max_new_tokens: int,
) -> Tuple[List[int], List[bool]]:


    counts: List[int] = []
    hit_limits: List[bool] = []
    for ids in generated_ids:
        flat = ids.reshape(-1)
        if pad_token_id is None:
            count = int(flat.numel())
            hit_limit = count >= int(max_new_tokens)
        else:
            is_pad = flat.eq(int(pad_token_id))
            count = int((~is_pad).sum().item())
            hit_limit = int(flat.numel()) >= int(max_new_tokens) and not bool(is_pad.any().item())
        counts.append(count)
        hit_limits.append(hit_limit)
    return counts, hit_limits


class RawQwen3VLClient:


    _cache: Dict[Tuple[str, str, str], Tuple[Any, Any, torch.device, float, Dict[str, Any]]] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_path: Path,
        device_text: str,
        attention: str,
        image_max_side: int,
        max_new_tokens: int,
        prompt: str,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device_text = str(device_text)
        self.attention = choose_attention(self.device_text, attention)
        self.image_max_side = max(320, int(image_max_side))
        self.max_new_tokens = max(16, int(max_new_tokens))
        self.prompt = str(prompt)
        self.min_pixels = _MIN_PIXELS
        self.max_pixels = self.image_max_side * 28 * 28
        self._processor_lock = threading.Lock()
        (
            self.model,
            self.processor,
            self.input_device,
            self.load_seconds,
            self.runtime_info,
        ) = self._load_once()

    def _load_once(self) -> Tuple[Any, Any, torch.device, float, Dict[str, Any]]:
        key = (str(self.model_path), self.device_text, self.attention)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            if not self.model_path.is_dir():
                raise FileNotFoundError(f"模型目录不存在: {self.model_path}")

            try:
                from transformers import AutoProcessor
                try:
                    from transformers import Qwen3VLForConditionalGeneration

                    model_class = Qwen3VLForConditionalGeneration
                except ImportError:
                    from transformers import AutoModelForImageTextToText

                    model_class = AutoModelForImageTextToText
            except ImportError as exc:
                raise RuntimeError(
                    "当前 transformers 不支持 Qwen3-VL，请按 requirements.txt 安装或更新依赖。"
                ) from exc

            started = time.perf_counter()
            use_cuda = self.device_text.startswith("cuda") and torch.cuda.is_available()
            if use_cuda:
                runtime_device = torch.device(self.device_text)
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map: Any = {"": self.device_text}


                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
            else:
                runtime_device = torch.device("cpu")
                dtype = torch.float32
                device_map = None

            kwargs: Dict[str, Any] = {
                "local_files_only": True,
                "low_cpu_mem_usage": True,
                "dtype": dtype,
                "attn_implementation": self.attention,
            }
            if device_map is not None:
                kwargs["device_map"] = device_map

            model = model_class.from_pretrained(str(self.model_path), **kwargs)
            if device_map is None:
                model.to(runtime_device)
            model.eval()
            sanitize_generation_config(model)

            processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.padding_side = "left"
                if processor.tokenizer.pad_token_id is None:
                    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

            input_device = next(
                parameter.device
                for parameter in model.parameters()
                if parameter.device.type != "meta"
            )
            gpu_name = "cpu"
            capability = ""
            if input_device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(input_device)
                major, minor = torch.cuda.get_device_capability(input_device)
                capability = f"{major}.{minor}"
            runtime_info = {
                "requested_device": self.device_text,
                "input_device": str(input_device),
                "device_name": gpu_name,
                "cuda_capability": capability,
                "dtype": str(dtype),
                "attention": self.attention,
            }
            loaded = (
                model,
                processor,
                input_device,
                time.perf_counter() - started,
                runtime_info,
            )
            self._cache[key] = loaded
            return loaded

    def _prompt_for_event(self, event: EventPair) -> str:
        if event.comparison_path is not None:
            return self.prompt
        return (
            "这张图来自一段交通视频，是按时间排列的故事板。"
            "每个格子都是不同时间帧，时间顺序为从左到右、从上到下。"
            "请用1至2句话，先说明主要车辆发生了什么，再给出判断；"
            "不要逐格复述，也不要罗列无关车辆。"
            "若看到碰撞、侧滑或横向失控、车辆停止在道路中、或者冒烟等情况，请明确说出；"
            "若没有看到明确事件，直接说明。不要输出JSON。"
        )

    def _conversation(self, event: EventPair) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for path in event.input_paths:
            content.append(
                {
                    "type": "image",
                    "image": str(path),
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
        content.append({"type": "text", "text": self._prompt_for_event(event)})
        return [{"role": "user", "content": content}]

    def _prepare_batch_cpu(self, events: Sequence[EventPair]) -> Tuple[Any, float, List[int]]:


        if process_vision_info is None:
            raise RuntimeError(
                "缺少 qwen-vl-utils，请先执行: pip install 'qwen-vl-utils>=0.0.14'"
            )
        items = list(events)
        if not items:
            raise ValueError("events 不能为空")

        conversations = [self._conversation(event) for event in items]
        texts: List[str] = []
        all_images: List[Any] = []
        all_videos: List[Any] = []

        started = time.perf_counter()
        with self._processor_lock:
            for conversation in conversations:
                texts.append(
                    self.processor.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                image_inputs, video_inputs = process_vision_info(conversation)
                if image_inputs:
                    all_images.extend(image_inputs)
                if video_inputs:
                    all_videos.extend(video_inputs)

            inputs = self.processor(
                text=texts,
                images=all_images or None,
                videos=all_videos or None,
                padding=True,
                return_tensors="pt",
            )
        return inputs, time.perf_counter() - started, _actual_input_tokens(inputs)

    def _pin_cpu_inputs(self, inputs: Any) -> Any:


        if self.input_device.type != "cuda" or not hasattr(inputs, "items"):
            return inputs
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.device.type == "cpu":
                try:
                    inputs[key] = value.pin_memory()
                except (RuntimeError, TypeError):
                    pass
        return inputs

    def prepare_event_cpu(self, event: EventPair) -> PreparedEvent:
        inputs, elapsed, input_tokens = self._prepare_batch_cpu([event])
        inputs = self._pin_cpu_inputs(inputs)
        return PreparedEvent(
            event=event,
            inputs_cpu=inputs,
            preprocess_seconds=elapsed,
            processor_path="standalone_process_vision_info_single_pinned",
            input_tokens=input_tokens[0] if input_tokens else 0,
        )

    def prepare_batch_cpu(self, events: Sequence[EventPair]) -> PreparedBatch:
        items = tuple(events)
        if not items:
            raise ValueError("events 不能为空")
        image_counts = {len(event.input_paths) for event in items}
        if len(image_counts) != 1:
            raise ValueError("同一 batch 中 event 的输入图片数量必须一致")
        inputs, elapsed, input_tokens = self._prepare_batch_cpu(items)
        inputs = self._pin_cpu_inputs(inputs)
        return PreparedBatch(
            events=items,
            inputs_cpu=inputs,
            preprocess_seconds=elapsed,
            processor_path="standalone_process_vision_info_batch_pinned",
            input_tokens=tuple(int(value) for value in input_tokens),
        )

    def _move_to_device(self, inputs_cpu: Any) -> Tuple[Any, float]:
        started = time.perf_counter()
        try:
            inputs = inputs_cpu.to(self.input_device, non_blocking=True)
        except TypeError:
            inputs = inputs_cpu.to(self.input_device)
        if self.input_device.type == "cuda":
            torch.cuda.synchronize(self.input_device)
        return inputs, time.perf_counter() - started

    def _generate(self, inputs: Any) -> Tuple[Any, float]:
        if self.input_device.type == "cuda":
            torch.cuda.synchronize(self.input_device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if self.input_device.type == "cuda":
            torch.cuda.synchronize(self.input_device)
        return output_ids, time.perf_counter() - started

    def infer_prepared(
        self,
        prepared: PreparedEvent,
        *,
        pipeline_wait_seconds: float = 0.0,
    ) -> RawResult:
        wall_started = time.perf_counter()
        inputs, transfer_seconds = self._move_to_device(prepared.inputs_cpu)
        output_ids, generate_seconds = self._generate(inputs)

        generated_ids = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        output_token_counts, hit_limits = _actual_generated_tokens(
            generated_ids, pad_token_id, self.max_new_tokens
        )
        output_tokens = output_token_counts[0] if output_token_counts else 0
        hit_limit = hit_limits[0] if hit_limits else False
        decode_started = time.perf_counter()
        with self._processor_lock:
            answers = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        decode_seconds = time.perf_counter() - decode_started
        raw_text = str(answers[0] if answers else "").strip()
        wall_seconds = time.perf_counter() - wall_started + prepared.preprocess_seconds

        del inputs, output_ids, generated_ids
        return RawResult(
            event=prepared.event,
            raw_text=raw_text,
            preprocess_seconds=prepared.preprocess_seconds,
            transfer_seconds=transfer_seconds,
            generate_seconds=generate_seconds,
            decode_seconds=decode_seconds,
            pipeline_wait_seconds=max(0.0, float(pipeline_wait_seconds)),
            wall_seconds=wall_seconds,
            batch_size=1,
            batch_wall_seconds=wall_seconds,
            timing_mode="per_event_exact_prefetch" if pipeline_wait_seconds >= 0 else "per_event_exact",
            processor_path=prepared.processor_path,
            input_tokens=prepared.input_tokens,
            output_tokens=output_tokens,
            hit_max_new_tokens=hit_limit,
            batch_preprocess_seconds=prepared.preprocess_seconds,
            batch_transfer_seconds=transfer_seconds,
            batch_generate_seconds=generate_seconds,
            batch_decode_seconds=decode_seconds,
            batch_total_output_tokens=output_tokens,
            batch_max_output_tokens=output_tokens,
        )

    def infer_event(self, event: EventPair) -> RawResult:
        prepared = self.prepare_event_cpu(event)
        return self.infer_prepared(prepared, pipeline_wait_seconds=-1.0)

    def infer_prepared_batch(self, prepared: PreparedBatch) -> List[RawResult]:
        items = list(prepared.events)
        if not items:
            return []
        if len(items) == 1:
            event_prepared = PreparedEvent(
                event=items[0],
                inputs_cpu=prepared.inputs_cpu,
                preprocess_seconds=prepared.preprocess_seconds,
                processor_path=prepared.processor_path,
                input_tokens=prepared.input_tokens[0] if prepared.input_tokens else 0,
            )
            return [self.infer_prepared(event_prepared, pipeline_wait_seconds=0.0)]

        batch_started = time.perf_counter()
        inputs, transfer_total = self._move_to_device(prepared.inputs_cpu)
        output_ids, generate_total = self._generate(inputs)
        generated_ids = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        output_token_counts, hit_limits = _actual_generated_tokens(
            generated_ids, pad_token_id, self.max_new_tokens
        )
        decode_started = time.perf_counter()
        with self._processor_lock:
            answers = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        decode_total = time.perf_counter() - decode_started


        batch_wall = time.perf_counter() - batch_started
        count = len(items)
        total_output_tokens = int(sum(output_token_counts))
        max_output_tokens = int(max(output_token_counts, default=0))

        results = [
            RawResult(
                event=event,
                raw_text=str(answer).strip(),
                preprocess_seconds=prepared.preprocess_seconds / count,
                transfer_seconds=transfer_total / count,
                generate_seconds=generate_total / count,
                decode_seconds=decode_total / count,
                wall_seconds=batch_wall,
                pipeline_wait_seconds=0.0,
                batch_size=count,
                batch_wall_seconds=batch_wall,
                timing_mode="batch_amortized_prefetched",
                processor_path=prepared.processor_path,
                input_tokens=(
                    prepared.input_tokens[index]
                    if index < len(prepared.input_tokens)
                    else 0
                ),
                output_tokens=(
                    output_token_counts[index]
                    if index < len(output_token_counts)
                    else 0
                ),
                hit_max_new_tokens=(
                    hit_limits[index] if index < len(hit_limits) else False
                ),
                batch_preprocess_seconds=prepared.preprocess_seconds,
                batch_transfer_seconds=transfer_total,
                batch_generate_seconds=generate_total,
                batch_decode_seconds=decode_total,
                batch_total_output_tokens=total_output_tokens,
                batch_max_output_tokens=max_output_tokens,
            )
            for index, (event, answer) in enumerate(zip(items, answers))
        ]
        del inputs, output_ids, generated_ids
        return results

    def infer_batch(self, events: Sequence[EventPair]) -> List[RawResult]:
        items = list(events)
        if not items:
            return []
        prepared = self.prepare_batch_cpu(items)
        return self.infer_prepared_batch(prepared)


LocalQwen3VLClient = RawQwen3VLClient


def result_path(event: EventPair, output_label: str) -> Path:
    return event.event_prefix.with_name(
        event.event_prefix.name + f"_vlm_{output_label}.json"
    )


def save_event_result(
    result: RawResult,
    output_path: Path,
    prompt: str,
    prompt_id: str,
    tag: str,
    model_path: Path,
    image_max_side: int,
    max_new_tokens: int,
) -> None:
    payload = {
        "status": "ok",
        "case": result.event.case_name,
        "segment": result.event.segment_name,
        "event": result.event.event_name,
        "input_images": [str(path) for path in result.event.input_paths],
        "metadata_path": (
            str(result.event.metadata_path)
            if result.event.metadata_path is not None
            else None
        ),
        "prompt_tag": tag,
        "prompt_hash": prompt_id,
        "prompt": prompt,
        "raw_text": result.raw_text,
        "token_counts": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "hit_max_new_tokens": result.hit_max_new_tokens,
            "batch_total_output_tokens": result.batch_total_output_tokens,
            "batch_max_output_tokens": result.batch_max_output_tokens,
        },
        "timing": {
            "preprocess_seconds": result.preprocess_seconds,
            "pipeline_wait_seconds": result.pipeline_wait_seconds,
            "transfer_seconds": result.transfer_seconds,
            "generate_seconds": result.generate_seconds,
            "decode_seconds": result.decode_seconds,
            "total_seconds": result.total_seconds,
            "pipeline_blocking_seconds": result.pipeline_blocking_seconds,
            "wall_seconds": result.wall_seconds,
            "batch_size": result.batch_size,
            "batch_wall_seconds": result.batch_wall_seconds,
            "batch_preprocess_seconds": result.batch_preprocess_seconds,
            "batch_transfer_seconds": result.batch_transfer_seconds,
            "batch_generate_seconds": result.batch_generate_seconds,
            "batch_decode_seconds": result.batch_decode_seconds,
            "batch_output_tokens_per_second": result.batch_output_tokens_per_second,
            "timing_mode": result.timing_mode,
            "note": (
                "batch_size>1 时 preprocess/generate/decode/total 为 batch 总时间除以样本数的吞吐均摊值；"
                "单条真实 latency 应查看 batch_wall_seconds。"
            ),
        },
        "processor_path": result.processor_path,
        "model_path": str(model_path),
        "image_max_side": int(image_max_side),
        "max_new_tokens": int(max_new_tokens),
        "postprocessing": "none",
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _result_row(result: RawResult, output_label: str) -> Dict[str, Any]:
    event = result.event
    return {
        "case": event.case_name,
        "segment": event.segment_name,
        "event": event.event_name,
        "comparison_image": str(event.comparison_path) if event.comparison_path else "",
        "storyboard_image": str(event.storyboard_path),
        "raw_text": result.raw_text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "hit_max_new_tokens": result.hit_max_new_tokens,
        "preprocess_seconds": f"{result.preprocess_seconds:.4f}",
        "pipeline_wait_seconds": f"{result.pipeline_wait_seconds:.4f}",
        "transfer_seconds": f"{result.transfer_seconds:.4f}",
        "generate_seconds": f"{result.generate_seconds:.4f}",
        "decode_seconds": f"{result.decode_seconds:.4f}",
        "total_seconds": f"{result.total_seconds:.4f}",
        "pipeline_blocking_seconds": f"{result.pipeline_blocking_seconds:.4f}",
        "wall_seconds": f"{result.wall_seconds:.4f}",
        "batch_size": result.batch_size,
        "batch_wall_seconds": f"{result.batch_wall_seconds:.4f}",
        "batch_generate_seconds": f"{result.batch_generate_seconds:.4f}",
        "batch_total_output_tokens": result.batch_total_output_tokens,
        "batch_max_output_tokens": result.batch_max_output_tokens,
        "batch_output_tokens_per_second": f"{result.batch_output_tokens_per_second:.2f}",
        "timing_mode": result.timing_mode,
        "processor_path": result.processor_path,
        "result_json": str(result_path(event, output_label)),
    }


def write_summary_csv(path: Path, results: Sequence[RawResult], output_label: str) -> None:
    fields = [
        "case", "segment", "event", "comparison_image", "storyboard_image", "raw_text",
        "input_tokens", "output_tokens", "hit_max_new_tokens", "preprocess_seconds", "pipeline_wait_seconds",
        "transfer_seconds", "generate_seconds", "decode_seconds", "total_seconds",
        "pipeline_blocking_seconds", "wall_seconds", "batch_size", "batch_wall_seconds",
        "batch_generate_seconds", "batch_total_output_tokens", "batch_max_output_tokens",
        "batch_output_tokens_per_second", "timing_mode", "processor_path", "result_json",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(_result_row(result, output_label))


def write_summary_json(
    path: Path,
    run_root: Path,
    prompt: str,
    prompt_id: str,
    tag: str,
    model_path: Path,
    load_seconds: float,
    discovered_count: int,
    skipped_cached_count: int,
    results: Sequence[RawResult],
    warnings: Sequence[str],
    elapsed_seconds: float,
    inference_mode: str,
    runtime_info: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "run_root": str(Path(run_root).expanduser().resolve()),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "runtime": dict(runtime_info or {}),
        "prompt_tag": tag,
        "prompt_hash": prompt_id,
        "prompt": prompt,
        "postprocessing": "none",
        "inference_mode": inference_mode,
        "timing_note": (
            "batch_size>1 时 event 级阶段时间是 batch 总时间的均摊吞吐成本，"
            "不是该 event 独立运行的真实 latency。"
        ),
        "discovered_event_count": discovered_count,
        "new_inference_count": len(results),
        "skipped_cached_count": skipped_cached_count,
        "model_load_seconds": load_seconds,
        "elapsed_seconds": elapsed_seconds,
        "events_per_second": (len(results) / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "warnings": list(warnings),
        "results": [
            {
                "case": item.event.case_name,
                "segment": item.event.segment_name,
                "event": item.event.event_name,
                "input_images": [str(path) for path in item.event.input_paths],
                "raw_text": item.raw_text,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "hit_max_new_tokens": item.hit_max_new_tokens,
                "total_seconds": item.total_seconds,
                "batch_size": item.batch_size,
                "batch_wall_seconds": item.batch_wall_seconds,
                "batch_generate_seconds": item.batch_generate_seconds,
                "timing_mode": item.timing_mode,
                "result_json": str(result_path(item.event, f"{safe_tag(tag)}_{prompt_id}")),
            }
            for item in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_root(run_root: Path) -> Path:
    root = Path(run_root).expanduser().resolve()
    if root.name == "adaptive_spatiotemporal_crops":
        return root.parent
    return root


def _compatible_chunks(events: Sequence[EventPair], batch_size: int) -> Iterable[List[EventPair]]:


    size = max(1, int(batch_size))
    current: List[EventPair] = []
    current_count: Optional[int] = None
    for event in events:
        count = len(event.input_paths)
        if current and (len(current) >= size or count != current_count):
            yield current
            current = []
            current_count = None
        if not current:
            current_count = count
        current.append(event)
    if current:
        yield current


def _save_or_print_result(
    result: RawResult,
    *,
    output_label: str,
    prompt: str,
    prompt_id: str,
    tag: str,
    client: RawQwen3VLClient,
    image_max_side: int,
    max_new_tokens: int,
    print_answers: bool,
) -> None:
    save_event_result(
        result=result,
        output_path=result_path(result.event, output_label),
        prompt=prompt,
        prompt_id=prompt_id,
        tag=tag,
        model_path=client.model_path,
        image_max_side=image_max_side,
        max_new_tokens=max_new_tokens,
    )
    if print_answers:
        if result.batch_size > 1:
            timing = (
                f"amortized={result.total_seconds:.3f}s/event, "
                f"batch_generate={result.batch_generate_seconds:.3f}s, "
                f"batch_wall={result.batch_wall_seconds:.3f}s, "
                f"batch={result.batch_size}, out_tokens={result.output_tokens}, "
                f"hit_limit={int(result.hit_max_new_tokens)}"
            )
        else:
            timing = (
                f"preprocess={result.preprocess_seconds:.3f}s, "
                f"generate={result.generate_seconds:.3f}s, "
                f"decode={result.decode_seconds:.3f}s, total={result.total_seconds:.3f}s, "
                f"out_tokens={result.output_tokens}, hit_limit={int(result.hit_max_new_tokens)}"
            )
        tqdm.write(
            "\n"
            f"[{result.event.case_name}/{result.event.segment_name}/{result.event.event_name}]\n"
            f"{result.raw_text}\n"
            f"[TIME] {timing}\n"
        )


def _error_payload(
    event: EventPair,
    *,
    prompt: str,
    prompt_id: str,
    tag: str,
    error: Exception,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "case": event.case_name,
        "segment": event.segment_name,
        "event": event.event_name,
        "input_images": [str(path) for path in event.input_paths],
        "metadata_path": str(event.metadata_path) if event.metadata_path else None,
        "prompt_tag": tag,
        "prompt_hash": prompt_id,
        "prompt": prompt,
        "error": str(error),
        "postprocessing": "none",
    }


def _run_serial_or_prefetch(
    client: RawQwen3VLClient,
    events: Sequence[EventPair],
    *,
    prefetch_depth: int,
    on_result: Any,
    on_error: Any,
    progress: tqdm,
) -> List[RawResult]:
    items = list(events)
    if not items:
        return []
    depth = max(1, int(prefetch_depth))
    results: List[RawResult] = []

    if depth <= 1:
        for event in items:
            try:
                result = client.infer_event(event)
                on_result(result)
                results.append(result)
            except Exception as exc:
                on_error(event, exc)
            finally:
                progress.update(1)
        return results

    queue: Deque[Tuple[EventPair, Future[PreparedEvent]]] = deque()
    next_index = 0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-prefetch") as executor:
        while next_index < len(items) and len(queue) < depth:
            event = items[next_index]
            queue.append((event, executor.submit(client.prepare_event_cpu, event)))
            next_index += 1

        while queue:
            event, future = queue.popleft()
            wait_started = time.perf_counter()
            try:
                prepared = future.result()
                wait_seconds = time.perf_counter() - wait_started
                if next_index < len(items):
                    next_event = items[next_index]
                    queue.append((next_event, executor.submit(client.prepare_event_cpu, next_event)))
                    next_index += 1
                result = client.infer_prepared(prepared, pipeline_wait_seconds=wait_seconds)
                on_result(result)
                results.append(result)
            except Exception as exc:
                on_error(event, exc)
                if next_index < len(items):
                    next_event = items[next_index]
                    queue.append((next_event, executor.submit(client.prepare_event_cpu, next_event)))
                    next_index += 1
            finally:
                progress.update(1)
    return results


def _infer_batch_resilient(
    client: RawQwen3VLClient,
    events: Sequence[EventPair],
    warnings: List[str],
) -> Tuple[List[RawResult], List[Tuple[EventPair, Exception]]]:


    items = list(events)
    if not items:
        return [], []
    try:
        return client.infer_batch(items), []
    except Exception as exc:
        if len(items) == 1:
            return [], [(items[0], exc)]
        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        is_oom = "out of memory" in str(exc).lower()
        if is_oom and len(items) > 4:
            midpoint = 4
        else:
            midpoint = max(1, len(items) // 2)
        warnings.append(
            f"batch={len(items)} 失败，自动拆为 {midpoint}+{len(items)-midpoint}: {exc}"
        )
        left_results, left_errors = _infer_batch_resilient(client, items[:midpoint], warnings)
        right_results, right_errors = _infer_batch_resilient(client, items[midpoint:], warnings)
        return left_results + right_results, left_errors + right_errors


def _run_batched_prefetch(
    client: RawQwen3VLClient,
    chunks: Sequence[Sequence[EventPair]],
    *,
    prefetch_depth: int,
    warnings: List[str],
    on_result: Any,
    on_error: Any,
    progress: tqdm,
) -> List[RawResult]:


    batches = [list(chunk) for chunk in chunks if chunk]
    if not batches:
        return []
    depth = max(1, int(prefetch_depth))
    results: List[RawResult] = []

    if depth <= 1 or not (
        hasattr(client, "prepare_batch_cpu") and hasattr(client, "infer_prepared_batch")
    ):
        for chunk in batches:
            chunk_results, chunk_errors = _infer_batch_resilient(client, chunk, warnings)
            for item in chunk_results:
                on_result(item)
                results.append(item)
                progress.update(1)
            for event, exc in chunk_errors:
                on_error(event, exc)
                progress.update(1)
        return results

    queue: Deque[Tuple[List[EventPair], Future[PreparedBatch]]] = deque()
    next_index = 0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-batch-prefetch") as executor:
        while next_index < len(batches) and len(queue) < depth:
            chunk = batches[next_index]
            queue.append((chunk, executor.submit(client.prepare_batch_cpu, chunk)))
            next_index += 1

        while queue:
            chunk, future = queue.popleft()
            try:
                prepared = future.result()

                if next_index < len(batches):
                    next_chunk = batches[next_index]
                    queue.append(
                        (next_chunk, executor.submit(client.prepare_batch_cpu, next_chunk))
                    )
                    next_index += 1
                chunk_results = client.infer_prepared_batch(prepared)
                chunk_errors: List[Tuple[EventPair, Exception]] = []
            except Exception as exc:

                warnings.append(
                    f"预取 batch={len(chunk)} 失败，改用安全回退路径: {exc}"
                )
                chunk_results, chunk_errors = _infer_batch_resilient(client, chunk, warnings)
                if next_index < len(batches) and len(queue) < depth:
                    next_chunk = batches[next_index]
                    queue.append(
                        (next_chunk, executor.submit(client.prepare_batch_cpu, next_chunk))
                    )
                    next_index += 1

            for item in chunk_results:
                on_result(item)
                results.append(item)
                progress.update(1)
            for event, exc in chunk_errors:
                on_error(event, exc)
                progress.update(1)
    return results

def diagnose_existing_output(
    video_output_dir: Path,
    model_path: Path,
    device: str,
    image_max_side: int = 896,
    max_new_tokens: int = 96,
    attention: str = "sdpa",
    overwrite: bool = False,
    batch_size: int = 6,
    max_events_per_segment: int = 0,
    prompt_file: Optional[Path] = None,
    tag: str = "raw_prompt_test",
    allow_storyboard_only: bool = False,
    print_answers: bool = True,
    selected_cases: Sequence[str] = (),
    selected_segments: Sequence[str] = (),
    event_regex: str = "",
    limit: int = 0,
    prefetch_depth: int = 2,
) -> Dict[str, Any]:
    requested_batch_size = int(batch_size)
    if requested_batch_size < 1 or requested_batch_size > 8:
        raise ValueError("--vlm-batch-size 仅支持 1~8；RTX 5090/32GB 推荐 6，显存不足会自动回退")
    if int(max_events_per_segment) > 0:
        print("[INFO] max_events_per_segment 由 Crop 阶段控制；VLM 处理所有已生成 event。")

    prompt = load_prompt(prompt_file)
    prompt_id = prompt_hash(prompt)
    safe_output_tag = safe_tag(tag)
    output_label = f"{safe_output_tag}_{prompt_id}"

    events, warnings = discover_events(
        run_root=video_output_dir,
        selected_cases=selected_cases,
        selected_segments=selected_segments,
        event_regex=event_regex,
        allow_storyboard_only=allow_storyboard_only,
    )
    if int(limit) > 0:
        events = events[: int(limit)]

    pending: List[EventPair] = []
    skipped_cached = 0
    for event in events:
        output = result_path(event, output_label)
        if output.is_file() and not overwrite:
            skipped_cached += 1
        else:
            pending.append(event)

    summary_root = _summary_root(video_output_dir)
    summary_root.mkdir(parents=True, exist_ok=True)
    csv_path = summary_root / f"vlm_{output_label}_summary.csv"
    json_path = summary_root / f"vlm_{output_label}_summary.json"
    inference_mode = (
        f"batch{requested_batch_size}_standalone_compatible"
        if requested_batch_size > 1
        else ("one_event_prefetch_2x1" if int(prefetch_depth) > 1 else "one_event_serial")
    )

    if not pending:
        result = {
            "run_root": str(Path(video_output_dir).expanduser().resolve()),
            "prompt_hash": prompt_id,
            "prompt": prompt,
            "postprocessing": "none",
            "inference_mode": inference_mode,
            "discovered_event_count": len(events),
            "new_inference_count": 0,
            "skipped_cached_count": skipped_cached,
            "model_load_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "summary_csv": str(csv_path),
            "summary_json": str(json_path),
            "warnings": warnings,
            "results": [],
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not csv_path.exists():
            write_summary_csv(csv_path, [], output_label)
        return result

    client = RawQwen3VLClient(
        model_path=model_path,
        device_text=device,
        attention=attention,
        image_max_side=image_max_side,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
    )
    runtime_info = dict(getattr(client, "runtime_info", {}) or {})
    print(
        "[VLM-RUNTIME] "
        f"device={runtime_info.get('input_device', device)} | "
        f"name={runtime_info.get('device_name', 'unknown')} | "
        f"dtype={runtime_info.get('dtype', 'unknown')} | "
        f"attention={runtime_info.get('attention', attention)} | "
        f"batch={requested_batch_size} | max_new_tokens={max_new_tokens} | "
        f"prefetch={max(1, int(prefetch_depth))}"
    )
    if requested_batch_size > 1:
        print(
            "[VLM-TIMING] batch 模式显示吞吐均摊值；后台会预处理下一批。"
            "单批 GPU 阻塞时间请看 batch_wall_seconds。"
        )

    started = time.perf_counter()
    results: List[RawResult] = []
    progress = tqdm(
        total=len(pending),
        desc=f"{summary_root.name} Qwen3-VL raw",
        unit="event",
        dynamic_ncols=True,
    )

    def on_result(item: RawResult) -> None:
        _save_or_print_result(
            item,
            output_label=output_label,
            prompt=prompt,
            prompt_id=prompt_id,
            tag=safe_output_tag,
            client=client,
            image_max_side=image_max_side,
            max_new_tokens=max_new_tokens,
            print_answers=print_answers,
        )

    def on_error(event: EventPair, exc: Exception) -> None:
        error_path = result_path(event, output_label)
        error_path.write_text(
            json.dumps(
                _error_payload(
                    event,
                    prompt=prompt,
                    prompt_id=prompt_id,
                    tag=safe_output_tag,
                    error=exc,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        warning = f"推理失败 {event.case_name}/{event.segment_name}/{event.event_name}: {exc}"
        warnings.append(warning)
        tqdm.write(f"[WARN] {warning}")

    try:
        if requested_batch_size == 1:
            results = _run_serial_or_prefetch(
                client,
                pending,
                prefetch_depth=max(1, int(prefetch_depth)),
                on_result=on_result,
                on_error=on_error,
                progress=progress,
            )
        else:
            results = _run_batched_prefetch(
                client,
                list(_compatible_chunks(pending, requested_batch_size)),
                prefetch_depth=max(1, int(prefetch_depth)),
                warnings=warnings,
                on_result=on_result,
                on_error=on_error,
                progress=progress,
            )
    finally:
        progress.close()

    elapsed = time.perf_counter() - started
    write_summary_csv(csv_path, results, output_label)
    write_summary_json(
        path=json_path,
        run_root=video_output_dir,
        prompt=prompt,
        prompt_id=prompt_id,
        tag=safe_output_tag,
        model_path=client.model_path,
        load_seconds=client.load_seconds,
        discovered_count=len(events),
        skipped_cached_count=skipped_cached,
        results=results,
        warnings=warnings,
        elapsed_seconds=elapsed,
        inference_mode=inference_mode,
        runtime_info=runtime_info,
    )

    return {
        "run_root": str(Path(video_output_dir).expanduser().resolve()),
        "prompt_hash": prompt_id,
        "prompt": prompt,
        "postprocessing": "none",
        "inference_mode": inference_mode,
        "runtime": runtime_info,
        "discovered_event_count": len(events),
        "new_inference_count": len(results),
        "skipped_cached_count": skipped_cached,
        "model_load_seconds": client.load_seconds,
        "elapsed_seconds": elapsed,
        "events_per_second": (len(results) / elapsed) if elapsed > 0 else 0.0,
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "warnings": warnings,
        "results": [
            {
                "case": item.event.case_name,
                "segment": item.event.segment_name,
                "event": item.event.event_name,
                "raw_text": item.raw_text,
                "total_seconds": item.total_seconds,
                "batch_size": item.batch_size,
                "batch_wall_seconds": item.batch_wall_seconds,
                "output_tokens": item.output_tokens,
                "result_json": str(result_path(item.event, output_label)),
            }
            for item in results
        ],
    }
