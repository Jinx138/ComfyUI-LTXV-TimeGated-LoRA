"""
LTXV Semantic Segment Planner v1.3.0

Small upstream segment-boundary planner for PromptRelay-compatible timing.

It resolves the visible-frame count from an LTX video latent, converts normalized
boundary handles into exact integer segment lengths, and emits a CSV string
compatible with PromptRelay's segment_lengths input.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

LTX23_TEMPORAL_COMPRESSION = 8


def _infer_ltx_latent_frames(video_latent) -> Tuple[int, str]:
    if video_latent is None:
        raise ValueError("video_latent is required")
    samples = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
    if not torch.is_tensor(samples):
        raise ValueError("video_latent does not contain a tensor under 'samples'")
    if samples.ndim != 5:
        raise ValueError(
            f"video_latent must be a video-only LATENT with samples shaped [B,C,T,H,W]. "
            f"Got shape={tuple(samples.shape)}."
        )
    latent_frames = int(samples.shape[2])
    if latent_frames <= 0:
        raise ValueError(f"video_latent has invalid temporal dimension: shape={tuple(samples.shape)}")
    return latent_frames, f"video_latent shape={tuple(samples.shape)}"


def _resolve_timing(video_latent) -> Tuple[int, int, str]:
    latent_frames, source = _infer_ltx_latent_frames(video_latent)
    total_frames = (latent_frames - 1) * LTX23_TEMPORAL_COMPRESSION + 1
    return int(total_frames), int(latent_frames), source


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_prompt_relay_local_segments(local_prompts: str) -> List[str]:
    """PromptRelay-compatible local prompt split: A | B | C."""
    if local_prompts is None:
        return []
    return [p.strip() for p in str(local_prompts).split("|") if p.strip()]


def _resolve_segment_count(semantic_source: str, local_prompts: str, segment_count: int) -> Tuple[int, List[str], str]:
    prompts = _parse_prompt_relay_local_segments(local_prompts)
    if semantic_source == "local_prompts" and prompts:
        count = len(prompts)
        source = "local_prompts"
    else:
        count = _safe_int(segment_count, 3)
        source = "manual_count"
    count = max(1, min(6, int(count)))
    labels = prompts[:count]
    while len(labels) < count:
        labels.append(f"segment {len(labels) + 1}")
    return count, labels, source


def _resolve_weighted_segment_lengths(
    *,
    total_frames: int,
    segment_count: int,
    boundaries_norm: List[float],
    min_segment_frames: int = 1,
) -> Tuple[List[int], List[int], List[float]]:
    total_frames = int(total_frames)
    segment_count = max(1, min(6, int(segment_count)))
    min_segment_frames = max(0, int(min_segment_frames))

    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if segment_count <= 1:
        return [total_frames], [], []

    if min_segment_frames * segment_count > total_frames:
        min_segment_frames = 1 if segment_count <= total_frames else 0

    raw: List[float] = []
    for value in list(boundaries_norm or [])[: segment_count - 1]:
        raw.append(max(0.0, min(1.0, _safe_float(value, 0.0))))

    # Missing handles default to even spacing.
    while len(raw) < segment_count - 1:
        raw.append((len(raw) + 1) / float(segment_count))

    raw = sorted(raw[: segment_count - 1])
    proposed_frames = [int(round(x * total_frames)) for x in raw]

    boundaries: List[int] = []
    prev = 0
    for i, frame in enumerate(proposed_frames):
        remaining_after = segment_count - i - 1
        lo = prev + min_segment_frames
        hi = total_frames - remaining_after * min_segment_frames
        frame = max(lo, min(hi, int(frame)))
        boundaries.append(frame)
        prev = frame

    edges = [0] + boundaries + [total_frames]
    lengths = [int(edges[i + 1] - edges[i]) for i in range(segment_count)]
    resolved_norm = [round(b / float(total_frames), 6) for b in boundaries]
    return lengths, boundaries, resolved_norm


class LTXVSemanticSegmentPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT", {
                    "tooltip": "Video-only LTX LATENT used to resolve the rendered frame count. The planner emits PromptRelay-compatible segment_lengths in visible-frame units."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Only used for schedule_info timing display."
                }),
                "semantic_source": (["local_prompts", "manual_count"], {
                    "default": "local_prompts",
                    "tooltip": "local_prompts counts PromptRelay-style pipe-separated prompts. manual_count uses segment_count directly."
                }),
                "local_prompts": ("STRING", {
                    "default": "segment 1 | segment 2 | segment 3",
                    "multiline": True,
                    "tooltip": "PromptRelay-compatible local prompt text. Pipe-separated entries define the number of semantic segments."
                }),
                "segment_count": ("INT", {
                    "default": 3, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Used only when semantic_source=manual_count. Maximum 6 segments = 5 active boundary handles."
                }),
                "boundary_1": ("FLOAT", {
                    "default": 0.333, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Boundary handle 1, normalized 0..1. Rounded to a whole frame."
                }),
                "boundary_2": ("FLOAT", {
                    "default": 0.667, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Boundary handle 2, normalized 0..1. For 3 segments, 0.25/0.75 gives 1/4-1/2-1/4."
                }),
                "boundary_3": ("FLOAT", {
                    "default": 0.750, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Boundary handle 3, active for 4+ segments."
                }),
                "boundary_4": ("FLOAT", {
                    "default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Boundary handle 4, active for 5+ segments."
                }),
                "boundary_5": ("FLOAT", {
                    "default": 0.938, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Boundary handle 5, active for 6 segments."
                }),
                "min_segment_frames": ("INT", {
                    "default": 1, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Minimum length for each segment after rounding and clamping."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "LTXV_SEGMENT_DATA", "STRING")
    RETURN_NAMES = ("segment_lengths_out", "segment_data", "schedule_info")
    FUNCTION = "plan"
    CATEGORY = "LTXV/Timing"
    DESCRIPTION = (
        "v1.3.0: compact upstream PromptRelay-compatible semantic segment planner. "
        "Converts local prompt segment count and boundary handles into exact frame segment_lengths."
    )

    def plan(
        self,
        video_latent,
        fps: float,
        semantic_source: str,
        local_prompts: str,
        segment_count: int,
        boundary_1: float,
        boundary_2: float,
        boundary_3: float,
        boundary_4: float,
        boundary_5: float,
        min_segment_frames: int,
    ):
        total_frames, latent_frames, timing_source = _resolve_timing(video_latent)
        resolved_count, labels, count_source = _resolve_segment_count(semantic_source, local_prompts, segment_count)

        boundaries_norm_in = [
            _safe_float(boundary_1, 0.333),
            _safe_float(boundary_2, 0.667),
            _safe_float(boundary_3, 0.750),
            _safe_float(boundary_4, 0.875),
            _safe_float(boundary_5, 0.938),
        ]

        lengths, boundaries_frames, boundaries_norm = _resolve_weighted_segment_lengths(
            total_frames=total_frames,
            segment_count=resolved_count,
            boundaries_norm=boundaries_norm_in,
            min_segment_frames=int(_safe_int(min_segment_frames, 1)),
        )
        segment_lengths_out = ",".join(str(int(x)) for x in lengths)

        data: Dict = {
            "version": "LTXV Semantic Segment Planner v1.3.0",
            "source_node": "LTXV Semantic Segment Planner",
            "total_frames": int(total_frames),
            "latent_frames": int(latent_frames),
            "temporal_compression": int(LTX23_TEMPORAL_COMPRESSION),
            "timing_source": str(timing_source),
            "fps": float(fps),
            "duration_seconds": round(float(total_frames) / float(max(1e-6, fps)), 6),
            "semantic_source": str(semantic_source),
            "count_source": str(count_source),
            "segment_count": int(resolved_count),
            "segment_labels": labels,
            "segment_lengths": [int(x) for x in lengths],
            "segment_lengths_csv": segment_lengths_out,
            "boundary_frames": [int(x) for x in boundaries_frames],
            "boundary_norm": boundaries_norm,
            "boundary_norm_input": [float(x) for x in boundaries_norm_in],
            "min_segment_frames": int(_safe_int(min_segment_frames, 1)),
            "local_prompts": str(local_prompts or ""),
        }

        schedule_info = (
            f"v1.3.0 segments={resolved_count} lengths={segment_lengths_out} "
            f"boundaries={data['boundary_frames']} source={count_source} "
            f"frames={total_frames} latent={latent_frames} duration={data['duration_seconds']:.2f}s"
        )
        return (segment_lengths_out, data, schedule_info)


NODE_CLASS_MAPPINGS = {
    "LTXVSemanticSegmentPlanner": LTXVSemanticSegmentPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVSemanticSegmentPlanner": "LTXV Semantic Segment Planner",
}
