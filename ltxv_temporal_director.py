"""
LTXV Temporal Director v1.3.0

New development branch beside the legacy Time-Gated LoRA node:
- LTXV Schedule Sync: one upstream source of segment truth from pipe-separated prompts and optional boundary sliders.
- LTXV Prompt Relay Encode (Scheduled): adapter around Kijai PromptRelay _encode_relay, fed by segment_data.
- LTXV Time-Gated LoRA (Scheduled): segment_data-driven TimeGate with no manual_frames UI.
- LTXV Temporal Schedule Preview: schedule + segment text + envelope curve preview image.

The legacy LTXV Time-Gated LoRA node is intentionally left unchanged.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.utils

from .ltx23_time_gated_lora import (
    LOG_PREFIX,
    LTX23_TEMPORAL_COMPRESSION,
    RAMP_MODES,
    RuntimeLoRASpec,
    TemporalLoRAModelFunctionWrapper,
    _alpha_from_sd,
    _build_envelope_data,
    _find_lora_pairs,
    _frame_profile_to_latent_profile,
    _get_lora_path,
    _infer_ltx_latent_frames,
    _is_linear_lora_pair,
    _module_debug_shape,
    _plateau_frames,
    _resolve_module_name,
    _resolve_timing,
    _safe_float,
    _safe_int,
)

DIRECTOR_PREFIX = "[LTXV Temporal Director]"
DIRECTOR_SCHEMA = "ltxv_segment_data_v1"
MAX_DIRECTOR_SEGMENTS = 4


def _split_segments_raw(local_prompts: str) -> List[str]:
    if local_prompts is None:
        return [""]
    return [p.strip() for p in str(local_prompts).split("|")]


def _nonempty_segments(local_prompts: str) -> List[str]:
    return [p.strip() for p in _split_segments_raw(local_prompts) if p.strip()]

_SEGMENT_PREFIX_RE = re.compile(r"^\s*(?:\[\s*)?(?:segment|scene|shot)\s*[-_#:]?\s*\d+\s*(?:\]\s*)?[:.\-–—]?\s*", re.IGNORECASE)


def _strip_leading_segment_prefix(text: str) -> Tuple[str, bool]:
    """Remove LLM-generated labels like 'segment 1:' from the beginning.

    PromptRelay only needs the semantic text blocks separated by '|'. Keeping
    segment labels in the text makes them conditioning tokens and also clutters
    the schedule preview/report.
    """
    raw = str(text or "").strip()
    cleaned = _SEGMENT_PREFIX_RE.sub("", raw, count=1).strip()
    return (cleaned or raw, cleaned != raw)


# Strict marker patterns used only when no explicit pipe is present. The
# line-start variant tolerates labels without punctuation; the inline variant
# deliberately requires ':'/'-' so prose such as "in segment 3 the action..."
# is not mistaken for a boundary.
_SEGMENT_MARKER_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:\[\s*)?(?:segment|scene|shot)\s*[-_#]?\s*(\d+)\s*(?:\]\s*)?(?:[:.\-–—])?[ \t]*"
)
_SEGMENT_MARKER_INLINE_RE = re.compile(
    r"(?i)(?:\[\s*)?(?:segment|scene|shot)\s*[-_#]?\s*(\d+)\s*(?:\]\s*)?\s*[:\-–—]\s*"
)
_PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")


def _clean_segment_parts(parts: List[str]) -> Tuple[List[str], int]:
    cleaned_parts: List[str] = []
    stripped_prefixes = 0
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        cleaned, changed = _strip_leading_segment_prefix(text)
        if changed:
            stripped_prefixes += 1
        cleaned_parts.append(cleaned)
    return cleaned_parts, stripped_prefixes


def _segments_from_markers(text: str) -> Optional[List[str]]:
    """Recover 2-4 ordered segments from explicit LLM labels.

    Accepted examples include `segment 1:`, `Scene #2 -`, `[shot 3]:`, and
    labels at line starts without punctuation. To avoid accidental splitting,
    the first marker must begin the prompt and numbering must be exactly 1..N.
    """
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_SEGMENT_MARKER_LINE_RE.finditer(raw))

    # Some models emit all labels inline. Only use the looser inline search if
    # the line-oriented search did not already find a usable sequence.
    if len(matches) < 2:
        matches = list(_SEGMENT_MARKER_INLINE_RE.finditer(raw))

    if len(matches) < 2:
        return None
    if raw[: matches[0].start()].strip():
        return None

    numbers = [int(m.group(1)) for m in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        return None

    parts: List[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        part = raw[start:end].strip()
        if not part:
            return None
        parts.append(part)
    return parts


def _parse_local_prompt_segments(local_prompts: str, expected: Optional[int]) -> Tuple[List[str], List[str], List[str]]:
    """Parse prompt segments with tolerant automatic separator recovery.

    Precedence is deliberately deterministic:
      1. Existing `|` separators.
      2. Ordered `segment/scene/shot N` labels.
      3. Blank-line paragraphs, only when they form 2-4 segments and match an
         explicit expected count when one is configured.
      4. Single-segment fallback with a warning.
    """
    text = str(local_prompts or "").strip()
    warnings: List[str] = []
    normalizations: List[str] = []

    if "|" in text:
        raw_parts = _split_segments_raw(text)
        if any(not part.strip() for part in raw_parts):
            warnings.append("empty prompt segment detected; empty entries are ignored")
        segments, stripped = _clean_segment_parts(raw_parts)
        if stripped:
            normalizations.append(f"removed {stripped} leading segment label(s) such as 'segment 1:'")
        return segments, warnings, normalizations

    marker_parts = _segments_from_markers(text)
    if marker_parts is not None:
        segments, stripped = _clean_segment_parts(marker_parts)
        inserted = max(0, len(segments) - 1)
        normalizations.append(
            f"inserted {inserted} pipe separator(s) from {len(segments)} ordered segment/scene/shot label(s)"
        )
        # Labels are consumed by the split itself; report them as removed even
        # though _strip_leading_segment_prefix normally sees only clean text.
        normalizations.append(f"removed {len(segments)} leading segment label(s) such as 'segment 1:'")
        return segments, warnings, normalizations

    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    paragraph_count_ok = 2 <= len(paragraphs) <= MAX_DIRECTOR_SEGMENTS
    expected_ok = expected is None or len(paragraphs) == expected
    if paragraph_count_ok and expected_ok:
        segments, stripped = _clean_segment_parts(paragraphs)
        inserted = max(0, len(segments) - 1)
        normalizations.append(
            f"inserted {inserted} pipe separator(s) from {len(segments)} blank-line paragraph(s)"
        )
        if stripped:
            normalizations.append(f"removed {stripped} leading segment label(s) such as 'segment 1:'")
        return segments, warnings, normalizations

    if len(paragraphs) > MAX_DIRECTOR_SEGMENTS:
        warnings.append(
            f"no pipe separator detected and {len(paragraphs)} paragraphs found; automatic paragraph recovery is limited to {MAX_DIRECTOR_SEGMENTS} segments"
        )
    elif expected is not None and len(paragraphs) > 1 and len(paragraphs) != expected:
        warnings.append(
            f"no pipe separator detected; paragraph recovery found {len(paragraphs)} block(s), expected {expected}"
        )
    else:
        warnings.append(
            "no pipe separator and no unambiguous ordered segment labels or 2-4 blank-line paragraphs detected; using one local segment"
        )

    fallback, changed = _strip_leading_segment_prefix(text or "segment 1")
    if changed:
        normalizations.append("removed 1 leading segment label such as 'segment 1:'")
    return [fallback], warnings, normalizations


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> List[str]:
    """Simple PIL word wrapper with ellipsis on overflow."""
    text = " ".join(str(text or "").split())
    if not text:
        return []
    words = text.split(" ")
    lines: List[str] = []
    current = ""

    def width_of(t: str) -> int:
        try:
            box = draw.textbbox((0, 0), t, font=font)
            return int(box[2] - box[0])
        except Exception:
            return len(t) * 8

    for word in words:
        candidate = word if not current else current + " " + word
        if width_of(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and current:
        lines.append(current)

    # If there are leftover words, mark the last visible line with ellipsis.
    consumed = " ".join(lines).split()
    if len(consumed) < len(words) and lines:
        last = lines[-1]
        ell = "…"
        while width_of(last + ell) > max_width and len(last) > 1:
            last = last[:-1].rstrip()
        lines[-1] = last + ell
    return lines[:max_lines]


def _parse_expected_segments(value) -> Optional[int]:
    text = str(value or "auto").strip().lower()
    if text == "auto":
        return None
    try:
        return max(1, min(MAX_DIRECTOR_SEGMENTS, int(text)))
    except Exception:
        return None


def _even_lengths(total_frames: int, segment_count: int) -> Tuple[List[int], List[int]]:
    total_frames = int(total_frames)
    segment_count = max(1, min(MAX_DIRECTOR_SEGMENTS, int(segment_count)))
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    base = total_frames // segment_count
    remainder = total_frames % segment_count
    # Keep early segments stable and put rounding remainder at the end. This makes
    # 505/3 -> 168,168,169 like the current PromptRelay timeline example.
    lengths = [base] * segment_count
    lengths[-1] += remainder
    edges = [0]
    for length in lengths:
        edges.append(edges[-1] + int(length))
    return lengths, edges

def _custom_lengths_from_boundaries(total_frames: int, segment_count: int, boundaries: List[float]) -> Tuple[List[int], List[int], List[float], List[str]]:
    """Resolve normalized boundary sliders into valid exact frame lengths.

    Boundaries are intentionally coarse dramaturgic controls, not manual frame
    surgery. They are sorted/clamped and converted to full visible-frame segment
    lengths that always sum exactly to total_frames.
    """
    total_frames = int(total_frames)
    segment_count = max(1, min(MAX_DIRECTOR_SEGMENTS, int(segment_count)))
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if segment_count <= 1:
        return [total_frames], [0, total_frames], [], []

    notes: List[str] = []
    needed = segment_count - 1
    raw = []
    for b in list(boundaries or [])[:needed]:
        try:
            raw.append(float(b))
        except Exception:
            raw.append(0.0)
    while len(raw) < needed:
        raw.append((len(raw) + 1) / float(segment_count))

    clamped = [max(0.0, min(1.0, x)) for x in raw]
    if any(abs(a - b) > 1e-9 for a, b in zip(raw, clamped)):
        notes.append("boundary slider(s) clamped to 0..1")
    sorted_bounds = sorted(clamped)
    if any(abs(a - b) > 1e-9 for a, b in zip(clamped, sorted_bounds)):
        notes.append("boundary slider(s) sorted into ascending order")

    edges = [0]
    resolved_norms: List[float] = []
    for i, b in enumerate(sorted_bounds):
        proposed = int(round(total_frames * b))
        # Reserve at least one frame for this and every remaining segment.
        min_allowed = edges[-1] + 1
        remaining_boundaries = needed - i - 1
        max_allowed = total_frames - (remaining_boundaries + 1)
        edge = max(min_allowed, min(max_allowed, proposed))
        if edge != proposed:
            notes.append("boundary frame(s) adjusted to keep every segment non-empty")
        edges.append(edge)
        resolved_norms.append(edge / float(total_frames))
    edges.append(total_frames)
    lengths = [edges[i + 1] - edges[i] for i in range(segment_count)]
    return lengths, edges, resolved_norms, notes


def _promptrelay_sigma_from_epsilon(epsilon: float) -> float:
    eps = max(1e-6, min(0.999999, float(epsilon)))
    return float(1.0 / math.log(1.0 / eps))


def _relay_timing_from_epsilon(epsilon: float, temporal_stride: int) -> Dict:
    eps = max(1e-6, min(0.999999, float(epsilon)))
    stride = max(1, int(temporal_stride or LTX23_TEMPORAL_COMPRESSION))
    sigma = _promptrelay_sigma_from_epsilon(eps)
    raw_frames = float(sigma) * float(stride)
    # A tiny epsilon still means a very narrow shoulder, not a zero-width one.
    edge_frames = int(max(0, round(raw_frames)))
    return {
        "schema": "ltxv_relay_timing_v1",
        "version": "LTXV Temporal Director v1.3.0",
        "source_node": "LTXV Prompt Relay Encode (Scheduled)",
        "epsilon": float(eps),
        "sigma": float(sigma),
        "temporal_stride": int(stride),
        "edge_softness_frames_raw": float(raw_frames),
        "edge_softness_frames": int(edge_frames),
        "formula": "sigma = 1 / ln(1 / epsilon); edge_softness_frames = round(sigma * temporal_stride)",
    }


def _validate_segment_data(segment_data: Dict, *, total_frames: Optional[int] = None) -> Dict:
    if not isinstance(segment_data, dict):
        raise ValueError("segment_data must be connected and must be a dictionary produced by LTXV Schedule Sync")
    lengths = segment_data.get("segment_lengths")
    if lengths is None and segment_data.get("segment_lengths_csv"):
        lengths = [int(x.strip()) for x in str(segment_data["segment_lengths_csv"]).split(",") if x.strip()]
    if not lengths:
        raise ValueError("segment_data has no segment_lengths")
    lengths = [int(x) for x in lengths]
    if any(x <= 0 for x in lengths):
        raise ValueError(f"segment_data contains non-positive segment length(s): {lengths}")
    edges = segment_data.get("segment_edges")
    if edges is None:
        edges = [0]
        for length in lengths:
            edges.append(edges[-1] + int(length))
    edges = [int(x) for x in edges]
    if len(edges) != len(lengths) + 1:
        raise ValueError(f"segment_edges length mismatch: edges={edges} lengths={lengths}")
    resolved_total = int(edges[-1])
    if sum(lengths) != resolved_total:
        raise ValueError(f"segment_lengths do not match segment_edges: lengths={lengths} edges={edges}")
    if total_frames is not None and int(total_frames) != resolved_total:
        raise ValueError(
            f"segment_data timeline mismatch: segment_data total={resolved_total}, connected video_latent total={int(total_frames)}. "
            "Use the same video_latent for LTXV Schedule Sync and this node."
        )
    labels = list(segment_data.get("segment_texts") or segment_data.get("segment_labels") or [])
    labels = [str(x) for x in labels[: len(lengths)]]
    while len(labels) < len(lengths):
        labels.append(f"segment {len(labels) + 1}")
    out = dict(segment_data)
    out["segment_lengths"] = lengths
    out["segment_lengths_csv"] = ",".join(str(x) for x in lengths)
    out["segment_edges"] = edges
    out["segment_texts"] = labels
    out["segment_count"] = len(lengths)
    out["total_frames"] = resolved_total
    fps = float(out.get("fps", 24.0) or 24.0)
    fps = max(1e-6, fps)
    out["duration_seconds"] = round(resolved_total / fps, 6)
    out["segment_time_edges_seconds"] = [round(edge / fps, 6) for edge in edges]
    out["segment_durations_seconds"] = [round(length / fps, 6) for length in lengths]
    return out


def _truncate(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)].rstrip() + "…"


class LTXVScheduleSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT", {
                    "tooltip": "Video-only LTX LATENT used to resolve exact visible-frame length. Use the same latent for downstream TimeGate/PromptRelay Scheduled nodes."
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "local_prompts": ("STRING", {
                    "default": "segment 1 action | segment 2 action | segment 3 action",
                    "multiline": True,
                    "tooltip": "Semantic prompts. Existing | separators are preferred; when absent, ordered segment/scene/shot labels or 2-4 blank-line paragraphs are converted automatically."
                }),
                "boundary_mode": (["equal", "custom"], {
                    "default": "equal",
                    "tooltip": "equal: distribute detected segments evenly. custom: use boundary_1..boundary_3 sliders / boundary bar as normalized timeline cuts."
                }),
                "boundary_1": ("FLOAT", {"default": 0.333, "min": 0.001, "max": 0.999, "step": 0.001}),
                "boundary_2": ("FLOAT", {"default": 0.667, "min": 0.001, "max": 0.999, "step": 0.001}),
                "boundary_3": ("FLOAT", {"default": 0.750, "min": 0.001, "max": 0.999, "step": 0.001}),
                "expected_segments": (["auto", "1", "2", "3", "4"], {
                    "default": "auto",
                    "tooltip": "Optional segment-count sanity check after automatic separator recovery. Temporal Director supports up to four segments."
                }),
                "separator_check": (["warn", "error"], {
                    "default": "warn",
                    "tooltip": "warn: keep running and report ambiguous parsing. error: stop when recovery or expected segment-count checks fail."
                }),
            }
        }

    RETURN_TYPES = ("LTXV_SEGMENT_DATA", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("segment_data", "segment_lengths_csv", "local_prompts_clean", "schedule_info")
    FUNCTION = "sync"
    CATEGORY = "LTXV/Temporal Director"
    DESCRIPTION = "v1.3.0: upstream schedule master with automatic separator recovery from segment labels or blank-line paragraphs; max 4 segments."

    def sync(self, video_latent, fps: float, local_prompts: str, boundary_mode: str, boundary_1: float, boundary_2: float, boundary_3: float, expected_segments: str, separator_check: str):
        total_frames, latent_frames, timing_source = _resolve_timing(video_latent)
        expected = _parse_expected_segments(expected_segments)
        segments, warnings, normalizations = _parse_local_prompt_segments(local_prompts, expected)
        if not segments:
            segments = ["segment 1"]
            warnings.append("no non-empty local prompt segment detected; using a single fallback segment")
        if len(segments) > MAX_DIRECTOR_SEGMENTS:
            raise ValueError(
                f"Temporal Director supports max {MAX_DIRECTOR_SEGMENTS} segments, detected {len(segments)}. "
                "Use at most three pipe separators, or use the classic TimeGate workflow for more complex timing."
            )

        if expected is not None and len(segments) != expected:
            warnings.append(f"expected {expected} segment(s), detected {len(segments)}")

        if separator_check == "error" and warnings:
            raise ValueError("LTXV Schedule Sync separator check failed: " + "; ".join(warnings))

        boundary_mode = str(boundary_mode or "equal")
        boundary_values_in = [float(boundary_1), float(boundary_2), float(boundary_3)]
        boundary_values_used: List[float] = []
        if boundary_mode == "custom" and len(segments) > 1:
            lengths, edges, boundary_values_used, boundary_notes = _custom_lengths_from_boundaries(total_frames, len(segments), boundary_values_in)
            normalizations.extend(boundary_notes)
        else:
            if boundary_mode not in ("equal", "custom"):
                normalizations.append(f"invalid boundary_mode '{boundary_mode}', using equal")
                boundary_mode = "equal"
            lengths, edges = _even_lengths(total_frames, len(segments))
            boundary_values_used = [edges[i] / float(total_frames) for i in range(1, len(edges) - 1)]

        csv = ",".join(str(int(x)) for x in lengths)
        clean = " | ".join(segments)
        duration = float(total_frames) / float(max(1e-6, fps))
        data = {
            "schema": DIRECTOR_SCHEMA,
            "version": "LTXV Temporal Director v1.3.0",
            "source_node": "LTXV Schedule Sync",
            "total_frames": int(total_frames),
            "latent_frames": int(latent_frames),
            "temporal_compression": int(LTX23_TEMPORAL_COMPRESSION),
            "timing_source": str(timing_source),
            "fps": float(fps),
            "duration_seconds": round(duration, 6),
            "segment_count": int(len(segments)),
            "segment_lengths": [int(x) for x in lengths],
            "segment_lengths_csv": csv,
            "segment_edges": [int(x) for x in edges],
            "segment_time_edges_seconds": [round(int(x) / float(max(1e-6, fps)), 6) for x in edges],
            "segment_durations_seconds": [round(int(x) / float(max(1e-6, fps)), 6) for x in lengths],
            "boundary_mode": str(boundary_mode),
            "boundary_values_input": [float(x) for x in boundary_values_in],
            "boundary_values_norm": [float(x) for x in boundary_values_used],
            "segment_texts": [str(x) for x in segments],
            "local_prompts_clean": clean,
            "expected_segments": str(expected_segments),
            "separator_check": str(separator_check),
            "warnings": list(warnings),
            "normalizations": list(normalizations),
            "valid": True,
        }
        info = (
            f"v1.3.0 ScheduleSync duration={duration:.3f}s frames={total_frames} latent={latent_frames} "
            f"segments={len(segments)} lengths={csv} boundary_mode={boundary_mode} "
            f"warnings={len(warnings)} normalizations={len(normalizations)}"
        )
        return (data, csv, clean, info)


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_promptrelay_shim(prompt_relay_mod, patches_mod):
    def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
        if not pixel_lengths:
            return []
        total_pixel = sum(pixel_lengths)
        if total_pixel <= 0:
            return [1] * len(pixel_lengths)
        naive_total = max(1, round(total_pixel / temporal_stride))
        target_total = min(latent_frames, naive_total)
        if target_total >= latent_frames - 1:
            target_total = latent_frames
        exact = [p * target_total / total_pixel for p in pixel_lengths]
        result = [int(e) for e in exact]
        diff = target_total - sum(result)
        if diff > 0:
            order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
            for k in range(diff):
                result[order[k % len(order)]] += 1
        for i in range(len(result)):
            if result[i] < 1:
                max_idx = max(range(len(result)), key=lambda j: result[j])
                if result[max_idx] > 1:
                    result[max_idx] -= 1
                    result[i] = 1
        return result

    def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None):
        for name, val in (("global_prompt", global_prompt), ("local_prompts", local_prompts), ("segment_lengths", segment_lengths)):
            if val is None:
                raise ValueError(f"PromptRelay Scheduled: '{name}' arrived as None")
        locals_list = [p.strip() for p in str(local_prompts).split("|") if p.strip()]
        if not locals_list:
            raise ValueError("At least one local prompt is required (separate with |)")
        arch, patch_size, temporal_stride = patches_mod.detect_model_type(model)
        samples = latent["samples"]
        latent_frames = samples.shape[2]
        tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
        parsed_lengths = None
        if str(segment_lengths).strip():
            pixel_lengths = [int(x.strip()) for x in str(segment_lengths).split(",") if x.strip()]
            parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)
        raw_tokenizer = prompt_relay_mod.get_raw_tokenizer(clip)
        full_prompt, token_ranges = prompt_relay_mod.map_token_indices(raw_tokenizer, str(global_prompt or ""), locals_list)
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        effective_lengths = prompt_relay_mod.distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
        q_token_idx = prompt_relay_mod.build_segments(token_ranges, effective_lengths, float(epsilon), relay_options)
        mask_fn = prompt_relay_mod.create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)
        patched = model.clone()
        patches_mod.apply_patches(patched, arch, mask_fn)
        relay_timing = _relay_timing_from_epsilon(float(epsilon), int(temporal_stride))
        relay_timing.update({
            "detected_model_arch": str(arch),
            "patch_size": tuple(patch_size),
            "latent_frames": int(latent_frames),
            "tokens_per_frame": int(tokens_per_frame),
            "promptrelay_effective_latent_lengths": [int(x) for x in effective_lengths],
        })
        return patched, conditioning, relay_timing

    return types.SimpleNamespace(_encode_relay=_encode_relay, _ltxv_promptrelay_shim=True)


def _find_promptrelay_module():
    # v1.3.0: deliberately avoid broad sys.modules probing. Some installed
    # extensions expose lazy modules whose __getattr__ raises ImportError for
    # optional native kernels (observed: seedvr2 _C_flashattention). A generic
    # hasattr(mod, "_encode_relay") scan can therefore fail before we ever reach
    # PromptRelay.
    #
    # Instead, find the PromptRelay folder explicitly and adapt only its two
    # lightweight helper modules. We still do not import PromptRelay/nodes.py.
    here = Path(__file__).resolve().parent
    custom_nodes_dir = here.parent
    candidates = []
    for pattern in ("ComfyUI-PromptRelay*", "*PromptRelay*"):
        candidates.extend(sorted(custom_nodes_dir.glob(pattern)))

    seen = set()
    errors = []
    for folder in candidates:
        folder = folder.resolve()
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        prompt_relay_py = folder / "prompt_relay.py"
        patches_py = folder / "patches.py"
        if not prompt_relay_py.exists() or not patches_py.exists():
            continue
        try:
            base = f"_ltxv_promptrelay_{abs(hash(str(folder))) & 0xffffffff:x}"
            prompt_relay_mod = _load_module_from_path(base + "_prompt_relay", prompt_relay_py)
            patches_mod = _load_module_from_path(base + "_patches", patches_py)
            return _make_promptrelay_shim(prompt_relay_mod, patches_mod)
        except Exception as exc:
            errors.append(f"{folder.name}: {type(exc).__name__}: {exc}")
            continue
    detail = "; ".join(errors[:5]) if errors else "no PromptRelay folder with prompt_relay.py and patches.py found"
    raise RuntimeError(
        "Kijai ComfyUI-PromptRelay could not be adapted. Install/enable ComfyUI-PromptRelay, restart ComfyUI, "
        f"then run LTXV Prompt Relay Encode (Scheduled) again. Details: {detail}"
    )


class LTXVPromptRelayEncodeScheduled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent": ("LATENT", {"tooltip": "The same empty/video latent that PromptRelay normally receives; dimensions are read from its shape."}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "segment_data": ("LTXV_SEGMENT_DATA", {"tooltip": "From LTXV Schedule Sync. Provides local prompts and exact visible-frame segment lengths."}),
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.99, "step": 0.0001}),
            },
            "optional": {
                "relay_options": ("RELAY_OPTIONS", {"tooltip": "Optional: connect Kijai Prompt Relay Advanced Options."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING", "LTXV_RELAY_TIMING")
    RETURN_NAMES = ("model", "positive", "schedule_info", "relay_timing")
    FUNCTION = "encode"
    CATEGORY = "LTXV/Temporal Director"
    DESCRIPTION = "v1.3.0: PromptRelay adapter without timeline UI. Emits relay_timing so TimeGate q-curves can share PromptRelay epsilon softness."

    def encode(self, model, clip, latent, global_prompt: str, segment_data: Dict, epsilon: float, relay_options=None):
        data = _validate_segment_data(segment_data)
        local_prompts = str(data.get("local_prompts_clean") or " | ".join(data.get("segment_texts") or []))
        segment_lengths = str(data.get("segment_lengths_csv") or ",".join(str(x) for x in data.get("segment_lengths", [])))
        if not local_prompts.strip():
            raise ValueError("LTXV Prompt Relay Encode (Scheduled): segment_data contains no local prompts")
        pr = _find_promptrelay_module()
        result = pr._encode_relay(
            model,
            clip,
            latent,
            str(global_prompt or ""),
            local_prompts,
            segment_lengths,
            float(epsilon),
            relay_options,
        )
        if isinstance(result, tuple) and len(result) >= 3:
            patched, conditioning, relay_timing = result[:3]
        else:
            patched, conditioning = result
            relay_timing = _relay_timing_from_epsilon(float(epsilon), LTX23_TEMPORAL_COMPRESSION)
        relay_timing = dict(relay_timing or {})
        relay_timing.update({
            "segment_count": int(data["segment_count"]),
            "segment_lengths": [int(x) for x in data["segment_lengths"]],
            "segment_lengths_csv": str(segment_lengths),
            "segment_edges": [int(x) for x in data["segment_edges"]],
            "total_frames": int(data["total_frames"]),
            "fps": float(data.get("fps", 24.0)),
        })
        info = (
            f"v1.3.0 PromptRelayScheduled segments={data['segment_count']} lengths={segment_lengths} "
            f"epsilon={float(relay_timing.get('epsilon', epsilon)):.6f} sigma={float(relay_timing.get('sigma', 0.0)):.4f} "
            f"edge_softness_frames={int(relay_timing.get('edge_softness_frames', 0))} source=LTXV_Schedule_Sync"
        )
        return (patched, conditioning, info, relay_timing)



def _director_q_mix(t: torch.Tensor, q: float) -> torch.Tensor:
    """Half-cosine shoulder curve for the Temporal Director.

    Unlike the legacy v1.1 power curve, the Director curve is intended to span
    across a schedule boundary. The boundary itself should be a soft shoulder,
    not the exact endpoint of the ramp. A half-cosine base gives a natural
    sine-like transition; q still controls the character:
      q = 1   : classic half-cosine / smooth S ramp
      q > 1   : front-loaded, reaches the target earlier
      q < 1   : delayed, but still approaches the target cleanly near the edge
    """
    q = max(float(q), 1e-6)
    x = torch.clamp(t, 0.0, 1.0)
    base = 0.5 - 0.5 * torch.cos(math.pi * x)
    if q >= 1.0:
        mix = 1.0 - torch.pow(1.0 - base, q)
    else:
        mix = torch.pow(base, 1.0 / q)
    return torch.clamp(mix, 0.0, 1.0)


def _fill_director_q_ramp(
    profile: torch.Tensor,
    start: int,
    end: int,
    value_a: float,
    value_b: float,
    ramp_q: float,
) -> None:
    """Fill a Director q-ramp over [start, end) using the half-cosine q mix."""
    total_frames = int(profile.numel())
    start = max(0, min(total_frames, int(start)))
    end = max(start, min(total_frames, int(end)))
    width = end - start
    if width <= 0:
        return
    if width == 1:
        profile[start:end] = float(value_b)
        return
    t = torch.linspace(0.0, 1.0, steps=width, dtype=torch.float32)
    mix = _director_q_mix(t, float(ramp_q))
    profile[start:end] = float(value_a) + (float(value_b) - float(value_a)) * mix


def _fill_director_q_ramp_virtual(
    profile: torch.Tensor,
    virtual_start: int,
    virtual_end: int,
    value_a: float,
    value_b: float,
    ramp_q: float,
    *,
    clip_start: int,
    clip_end: int,
) -> Tuple[int, int]:
    """Render the visible slice of a q-ramp whose full interval may lie outside the video.

    This is used at the first and last schedule segments. A missing neighboring
    segment is treated as virtual pre-roll/post-roll instead of discarding
    strength_before/strength_after. Clipping preserves the original ramp phase:
    a shoulder shifted partly outside the video is not compressed into the
    remaining visible frames.
    """
    total_frames = int(profile.numel())
    virtual_start = int(virtual_start)
    virtual_end = int(virtual_end)
    clip_start = max(0, min(total_frames, int(clip_start)))
    clip_end = max(clip_start, min(total_frames, int(clip_end)))
    width = virtual_end - virtual_start
    if width <= 0 or clip_end <= clip_start:
        return clip_start, clip_start

    visible_start = max(clip_start, virtual_start)
    visible_end = min(clip_end, virtual_end)
    if visible_end <= visible_start:
        return visible_start, visible_start

    if width == 1:
        profile[visible_start:visible_end] = float(value_b)
        return visible_start, visible_end

    indices = torch.arange(visible_start, visible_end, dtype=torch.float32)
    t = (indices - float(virtual_start)) / float(width - 1)
    mix = _director_q_mix(torch.clamp(t, 0.0, 1.0), float(ramp_q))
    profile[visible_start:visible_end] = float(value_a) + (float(value_b) - float(value_a)) * mix
    return visible_start, visible_end

def _build_scheduled_segment_profile(
    *,
    total_frames: int,
    segment_data: Dict,
    target_segment: int,
    strength_before: float,
    strength_during: float,
    strength_after: float,
    ramp_mode: str,
    q_in: float,
    q_out: float,
    q_in_shift: float = 0.0,
    q_out_shift: float = 0.0,
    edge_softness_frames: int = 0,
    relay_timing: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict]:
    data = _validate_segment_data(segment_data, total_frames=total_frames)
    count = int(data["segment_count"])
    edges = [int(x) for x in data["segment_edges"]]
    texts = list(data.get("segment_texts") or [])
    target_int = int(target_segment)
    if target_int < 1 or target_int > count:
        raise ValueError(f"target_segment={target_int} is outside the Schedule Sync segment range 1..{count}")
    target_index = target_int - 1
    region_start = int(edges[target_index])
    region_end = int(edges[target_index + 1])
    region_len = max(1, region_end - region_start)

    # A missing edge neighbor is represented by a full virtual segment with the
    # same duration as the target. This makes first/last-segment q-curves obey
    # exactly the same geometry as internal segments; the off-screen part is
    # merely clipped, never compressed or replaced by a special edge shoulder.
    incoming_edge_virtual = bool(target_index == 0)
    outgoing_edge_virtual = bool(target_index == count - 1)
    before_start = int(edges[target_index - 1]) if target_index > 0 else int(region_start - region_len)
    after_end = int(edges[target_index + 2]) if target_index + 2 < len(edges) else int(region_end + region_len)

    raw_edge_softness = max(0, int(round(float(edge_softness_frames or 0))))
    # The relay epsilon shoulder should only breathe a little into the active
    # segment. Most of the q-curve still lives in the neighboring segment.
    max_edge_softness = max(0, int(math.floor(region_len * 0.25)))
    resolved_edge_softness = min(raw_edge_softness, max_edge_softness) if ramp_mode == "q_curve" else 0

    q_in_shift = max(-1.0, min(1.0, float(q_in_shift or 0.0)))
    q_out_shift = max(-1.0, min(1.0, float(q_out_shift or 0.0)))

    # v1.3.0: q shifts control transition anchors, while the complete ramps
    # retain one unified neighbor-to-target / target-to-neighbor geometry.
    # The inward extrema meet at the target midpoint and can form a peak. The
    # outward extrema reach the midpoint of a real or virtual neighbor.
    region_mid = int(round((region_start + region_end) * 0.5))
    region_mid = max(region_start, min(region_end, region_mid))
    before_mid = int(round((before_start + region_start) * 0.5))
    after_mid = int(round((region_end + after_end) * 0.5))

    default_incoming_end = int(region_start + resolved_edge_softness)
    default_outgoing_start = int(region_end - resolved_edge_softness)
    phase_shift_active = bool(ramp_mode == "q_curve" and resolved_edge_softness > 0)

    if not phase_shift_active:
        # Preserve no-relay/no-softness behavior: shift widgets are inert until
        # PromptRelay timing supplies a usable epsilon shoulder.
        incoming_transition_frame = int(default_incoming_end)
        outgoing_transition_frame = int(default_outgoing_start)
    else:
        if q_in_shift >= 0.0:
            incoming_transition_frame = int(round(default_incoming_end + (region_mid - default_incoming_end) * q_in_shift))
        else:
            incoming_transition_frame = int(round(default_incoming_end + (before_mid - default_incoming_end) * (-q_in_shift)))

        if q_out_shift <= 0.0:
            outgoing_transition_frame = int(round(default_outgoing_start + (region_mid - default_outgoing_start) * (-q_out_shift)))
        else:
            outgoing_transition_frame = int(round(default_outgoing_start + (after_mid - default_outgoing_start) * q_out_shift))

    # Keep the two transitions ordered. At their inward extrema they may meet
    # exactly at region_mid, yielding a peak with no forced plateau.
    incoming_transition_frame = max(before_start + 1, min(region_mid, int(incoming_transition_frame)))
    outgoing_transition_frame = min(after_end - 1, max(region_mid, int(outgoing_transition_frame)))
    if outgoing_transition_frame < incoming_transition_frame:
        outgoing_transition_frame = incoming_transition_frame

    # Reported shift frames remain deltas from the zero/default phase position.
    q_in_shift_frames = int(incoming_transition_frame - default_incoming_end)
    q_out_shift_frames = int(outgoing_transition_frame - default_outgoing_start)
    incoming_shift_anchor = int(incoming_transition_frame)
    outgoing_shift_anchor = int(outgoing_transition_frame)

    profile = torch.empty((int(total_frames),), dtype=torch.float32)

    # Defaults used by flat mode and metadata.
    incoming_start = int(region_start)
    incoming_end = int(region_start)
    outgoing_start = int(region_end)
    outgoing_end = int(region_end)
    incoming_virtual_start = int(region_start)
    incoming_virtual_end = int(region_start)
    outgoing_virtual_start = int(region_end)
    outgoing_virtual_end = int(region_end)

    if ramp_mode == "flat":
        profile[:region_start] = float(strength_before)
        profile[region_start:region_end] = float(strength_during)
        if region_end < total_frames:
            profile[region_end:] = float(strength_after)
        incoming_edge_virtual = False
        outgoing_edge_virtual = False
    elif ramp_mode == "q_curve":
        # Unified full ramp intervals. For edge segments, before_start/after_end
        # lie outside the clip and _fill_director_q_ramp_virtual renders only
        # the visible slice at its original phase.
        incoming_virtual_start = int(before_start)
        incoming_virtual_end = int(incoming_transition_frame)
        outgoing_virtual_start = int(outgoing_transition_frame)
        outgoing_virtual_end = int(after_end)

        # Piecewise baseline: BEFORE -> incoming ramp -> DURING plateau/peak ->
        # outgoing ramp -> AFTER. This explicit DURING interval fixes the dev13
        # bug where strength_after took over immediately at region_end when an
        # outgoing transition was shifted into the following segment.
        profile.fill_(float(strength_before))

        incoming_start, incoming_end = _fill_director_q_ramp_virtual(
            profile,
            incoming_virtual_start,
            incoming_virtual_end,
            float(strength_before),
            float(strength_during),
            float(q_in),
            clip_start=0,
            clip_end=total_frames,
        )

        plateau_start = max(0, min(total_frames, int(incoming_transition_frame)))
        plateau_end = max(plateau_start, min(total_frames, int(outgoing_transition_frame)))
        if plateau_end > plateau_start:
            profile[plateau_start:plateau_end] = float(strength_during)

        outgoing_start, outgoing_end = _fill_director_q_ramp_virtual(
            profile,
            outgoing_virtual_start,
            outgoing_virtual_end,
            float(strength_during),
            float(strength_after),
            float(q_out),
            clip_start=0,
            clip_end=total_frames,
        )

        after_visible_start = max(0, min(total_frames, int(outgoing_virtual_end)))
        if after_visible_start < total_frames:
            profile[after_visible_start:] = float(strength_after)
    else:
        raise ValueError(f"Unknown ramp_mode: {ramp_mode}")

    relay_timing = dict(relay_timing or {}) if isinstance(relay_timing, dict) else {}

    meta = {
        "mode": "segment_data",
        "segment_data_schema": data.get("schema", ""),
        "segment_count": count,
        "target_segment": int(target_index + 1),
        "target_segment_text": texts[target_index] if target_index < len(texts) else f"segment {target_index + 1}",
        "segment_lengths": [int(x) for x in data["segment_lengths"]],
        "segment_lengths_csv": str(data["segment_lengths_csv"]),
        "segment_edges": [int(x) for x in edges],
        "segment_texts": [str(x) for x in texts],
        "region_start": int(region_start),
        "region_end": int(region_end),
        "region_len": int(region_len),
        "before_neighbor_start": int(before_start),
        "before_neighbor_end": int(region_start),
        "after_neighbor_start": int(region_end),
        "after_neighbor_end": int(after_end),
        "hold_strength": False,
        "local_baseline_strength": None,
        "q_in": float(q_in),
        "q_out": float(q_out),
        "q_in_shift": float(q_in_shift),
        "q_out_shift": float(q_out_shift),
        "q_in_shift_frames": int(q_in_shift_frames),
        "q_out_shift_frames": int(q_out_shift_frames),
        "incoming_shift_anchor_frame": int(incoming_shift_anchor),
        "outgoing_shift_anchor_frame": int(outgoing_shift_anchor),
        "incoming_transition_frame": int(incoming_transition_frame),
        "outgoing_transition_frame": int(outgoing_transition_frame),
        "target_segment_mid_frame": int(region_mid),
        "before_segment_mid_frame": int(before_mid),
        "after_segment_mid_frame": int(after_mid),
        "default_incoming_transition_frame": int(default_incoming_end),
        "default_outgoing_transition_frame": int(default_outgoing_start),
        "phase_shift_active": bool(phase_shift_active),
        "incoming_ramp_start_frame": int(incoming_start),
        "incoming_ramp_end_frame": int(incoming_end),
        "outgoing_ramp_start_frame": int(outgoing_start),
        "outgoing_ramp_end_frame": int(outgoing_end),
        "incoming_virtual_ramp_start_frame": int(incoming_virtual_start),
        "incoming_virtual_ramp_end_frame": int(incoming_virtual_end),
        "outgoing_virtual_ramp_start_frame": int(outgoing_virtual_start),
        "outgoing_virtual_ramp_end_frame": int(outgoing_virtual_end),
        "incoming_edge_virtual": bool(incoming_edge_virtual),
        "outgoing_edge_virtual": bool(outgoing_edge_virtual),
        "relay_epsilon": relay_timing.get("epsilon"),
        "relay_sigma": relay_timing.get("sigma"),
        "relay_temporal_stride": relay_timing.get("temporal_stride"),
        "edge_softness_frames_raw": relay_timing.get("edge_softness_frames_raw", float(raw_edge_softness)),
        "edge_softness_frames_requested": int(raw_edge_softness),
        "edge_softness_frames": int(resolved_edge_softness),
        "edge_softness_cap_frames": int(max_edge_softness),
        "edge_softness_source": "relay_timing" if relay_timing else "none",
        "ramp_q_formula": "Director half-cosine q-curve: base=0.5-0.5*cos(pi*t); q>1 front-loaded, q<1 delayed; relay softness defines the zero-position anchor; q_in_shift/q_out_shift move transition anchors between real/virtual neighboring midpoints and the target midpoint; full edge ramps continue outside the clip without compression; q_in=+1 plus q_out=-1 can meet at the target midpoint to form a peak",
        "warnings": list(data.get("warnings") or []),
    }
    return profile, meta

class LTXVTimeGatedLoRAScheduled:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MODEL after PromptRelay Scheduled. This scheduled TimeGate should remain the last temporal LoRA patch before sampler/guider."}),
                "video_latent": ("LATENT", {"tooltip": "Same video-only LTX LATENT used by LTXV Schedule Sync."}),
                "segment_data": ("LTXV_SEGMENT_DATA", {"tooltip": "From LTXV Schedule Sync. Determines exact segment boundaries."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "target_segment": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "One-based segment index from Schedule Sync. Temporal Director supports max four segments."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05}),
                "strength_before": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05, "tooltip": "Level before the incoming ramp. For target segment 1, q_curve uses a full virtual preceding segment and renders only the visible slice without compressing the curve."}),
                "strength_during": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05}),
                "strength_after": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05, "tooltip": "Level after the outgoing ramp. For the final target segment, q_curve uses a full virtual following segment and renders only the visible slice without compressing the curve."}),
                "ramp_mode": (RAMP_MODES, {"default": "q_curve"}),
                "q_in": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.05}),
                "q_out": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.05}),
                "memory_mode": (["optimized_safe", "optimized_low_vram_inplace"], {"default": "optimized_safe"}),
                "q_in_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Move the incoming transition phase. 0 keeps the epsilon-derived default. +1 completes the incoming ramp at the target segment midpoint; -1 moves completion to the preceding (or virtual pre-roll) midpoint. Use +1 with q_out_shift=-1 for a centered LoRA peak."}),
                "q_out_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Move the outgoing transition phase. 0 keeps the epsilon-derived default. -1 starts the outgoing ramp at the target segment midpoint; +1 delays it to the following (or virtual post-roll) midpoint. Use -1 with q_in_shift=+1 for a centered LoRA peak."}),
            },
            "optional": {
                "relay_timing": ("LTXV_RELAY_TIMING", {"tooltip": "Optional from LTXV Prompt Relay Encode (Scheduled). PromptRelay epsilon defines the default q-curve shoulder; q_in_shift/q_out_shift move the phase within the shared segment timeline."}),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT", "STRING", "LTXV_ENVELOPE_DATA")
    RETURN_NAMES = ("model", "video_latent", "schedule_info", "data")
    FUNCTION = "apply"
    CATEGORY = "LTXV/Temporal Director"
    DESCRIPTION = "v1.3.0: scheduled TimeGate with PromptRelay-coupled q-curves, explicit before/during/after phase holds, centered-peak shifts, and symmetric full virtual edge segments."

    def apply(
        self,
        model,
        video_latent,
        segment_data: Dict,
        fps: float,
        lora_name: str,
        target_segment: int,
        strength_model: float,
        strength_before: float,
        strength_during: float,
        strength_after: float,
        ramp_mode: str,
        q_in: float,
        q_out: float,
        memory_mode: str,
        q_in_shift: float = 0.0,
        q_out_shift: float = 0.0,
        relay_timing: Optional[Dict] = None,
    ):
        temporal_compression = LTX23_TEMPORAL_COMPRESSION
        resolved_frames, resolved_latent_frames, timing_source = _resolve_timing(video_latent)
        q_in = _safe_float(q_in, 1.0, "q_in")
        q_out = _safe_float(q_out, q_in, "q_out")
        q_in_shift = max(-1.0, min(1.0, _safe_float(q_in_shift, 0.0, "q_in_shift")))
        q_out_shift = max(-1.0, min(1.0, _safe_float(q_out_shift, 0.0, "q_out_shift")))
        if ramp_mode not in RAMP_MODES:
            logging.warning("%s invalid ramp_mode=%r; using q_curve", DIRECTOR_PREFIX, ramp_mode)
            ramp_mode = "q_curve"
        if memory_mode not in ("optimized_safe", "optimized_low_vram_inplace"):
            logging.warning("%s invalid memory_mode=%r; using optimized_safe", DIRECTOR_PREFIX, memory_mode)
            memory_mode = "optimized_safe"

        relay_timing_data = dict(relay_timing or {}) if isinstance(relay_timing, dict) else {}
        edge_softness_frames = int(relay_timing_data.get("edge_softness_frames", 0)) if ramp_mode == "q_curve" else 0

        frame_profile, envelope_meta = _build_scheduled_segment_profile(
            total_frames=resolved_frames,
            segment_data=segment_data,
            target_segment=_safe_int(target_segment, 1, "target_segment"),
            strength_before=float(strength_before),
            strength_during=float(strength_during),
            strength_after=float(strength_after),
            ramp_mode=ramp_mode,
            q_in=float(q_in),
            q_out=float(q_out),
            q_in_shift=float(q_in_shift),
            q_out_shift=float(q_out_shift),
            edge_softness_frames=edge_softness_frames,
            relay_timing=relay_timing_data,
        )
        latent_profile = _frame_profile_to_latent_profile(frame_profile, temporal_compression=temporal_compression)
        if int(latent_profile.numel()) != int(resolved_latent_frames):
            raise RuntimeError(
                f"Internal timing mismatch: profile latent_frames={int(latent_profile.numel())}, resolved_latent_frames={resolved_latent_frames}."
            )

        region_start = int(envelope_meta["region_start"])
        region_end = int(envelope_meta["region_end"])
        warnings = list(envelope_meta.get("warnings") or [])
        schedule_description = (
            f"segment_data target_segment={envelope_meta['target_segment']}/{envelope_meta['segment_count']} "
            f"frames=[{region_start},{region_end}) lengths={envelope_meta['segment_lengths_csv']} "
            f"strengths={float(strength_before):.3f},{float(strength_during):.3f},{float(strength_after):.3f} "
            f"ramp_mode={ramp_mode} q_in={float(q_in):.3f} q_out={float(q_out):.3f} "
            f"q_in_shift={float(q_in_shift):+.2f}/{int(envelope_meta.get('q_in_shift_frames', 0)):+d}f "
            f"q_out_shift={float(q_out_shift):+.2f}/{int(envelope_meta.get('q_out_shift_frames', 0)):+d}f "
            f"edge_softness_frames={int(envelope_meta.get('edge_softness_frames', 0))}"
        )
        envelope_data = _build_envelope_data(
            version="LTXV Time-Gated LoRA Scheduled v1.3.0",
            total_frames=resolved_frames,
            latent_frames=resolved_latent_frames,
            temporal_compression=temporal_compression,
            timing_source=timing_source,
            fps=float(fps),
            schedule_mode="segment_data",
            effect_region=f"segment {envelope_meta['target_segment']}",
            envelope_mode="local",
            strength_model=float(strength_model),
            strength_before=float(strength_before),
            strength_during=float(strength_during),
            strength_after=float(strength_after),
            ramp_mode=ramp_mode,
            q_in=float(q_in),
            q_out=float(q_out),
            transition_frames=0,
            segment_lengths=str(envelope_meta["segment_lengths_csv"]),
            segment_strengths="",
            memory_mode=memory_mode,
            frame_profile=frame_profile,
            latent_profile=latent_profile,
            meta=envelope_meta,
            warnings=warnings,
            schedule_description=schedule_description,
            lora_name=lora_name,
        )

        effective_profile = frame_profile * float(strength_model)
        effective_min = float(effective_profile.min().item())
        effective_max = float(effective_profile.max().item())
        full_strength_frames = _plateau_frames(frame_profile, float(strength_during))
        warning_text = "; ".join(warnings) if warnings else "none"

        if abs(float(strength_model)) < 1e-8 or float(frame_profile.abs().max()) < 1e-8:
            return (
                model,
                video_latent,
                f"v1.3.0 scheduled passthrough; {schedule_description}; frames={resolved_frames}; "
                f"latent_frames={resolved_latent_frames}; full_strength_frames={full_strength_frames}; warnings={warning_text}",
                envelope_data,
            )

        lora_path = _get_lora_path(lora_name)
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        pairs = _find_lora_pairs(sd)
        if not pairs:
            raise ValueError(f"No supported LoRA tensors found in {lora_name}.")

        out = model.clone()
        modules = dict(out.model.named_modules())
        stale_wrappers = [n for n, m in modules.items() if type(m).__name__ == "TemporalLoRALinearWrapper"]
        if stale_wrappers:
            raise RuntimeError(
                "Persistent TemporalLoRALinearWrapper objects from older builds are already present. Restart ComfyUI. "
                f"First stale layer: {stale_wrappers[0]}"
            )

        specs: List[RuntimeLoRASpec] = []
        skipped = []
        audio_skipped = 0
        for pair in pairs:
            module_name = _resolve_module_name(pair.module_hint, modules)
            if module_name is None:
                skipped.append((pair.module_hint, "target module not found"))
                continue
            if ".audio_" in module_name:
                audio_skipped += 1
                skipped.append((module_name, "audio layer omitted by design"))
                continue
            module = modules[module_name]
            down = sd[pair.down_key]
            up = sd[pair.up_key]
            if not _is_linear_lora_pair(down, up, module):
                skipped.append((module_name, f"unsupported shape/module down={tuple(down.shape)} up={tuple(up.shape)} module={type(module).__name__} {_module_debug_shape(module)}"))
                continue
            rank = int(down.shape[0])
            alpha = _alpha_from_sd(sd, pair, rank)
            specs.append(RuntimeLoRASpec(
                module_name=module_name,
                down=down.detach().clone().to(device="cpu", dtype=torch.float32),
                up=up.detach().clone().to(device="cpu", dtype=torch.float32),
                scale=alpha / float(rank),
                strength=float(strength_model),
                latent_profile=latent_profile.detach().clone().to(device="cpu", dtype=torch.float32),
            ))

        if not specs:
            details = "; ".join([f"{name}: {why}" for name, why in skipped[:20]])
            raise ValueError(f"No visual LoRA layers could be applied to this MODEL. First skipped entries: {details}")

        previous_wrapper = out.model_options.get("model_function_wrapper")
        runtime_wrapper = TemporalLoRAModelFunctionWrapper(out.model, specs, previous_wrapper, memory_mode=memory_mode)
        out.set_model_unet_function_wrapper(runtime_wrapper)

        info = (
            f"v1.3.0 scheduled; lora={lora_name}; {schedule_description}; frames={resolved_frames}; "
            f"latent_frames={resolved_latent_frames}; full_strength_frames={full_strength_frames}; "
            f"prepared_visual_layers={len(specs)}; audio_skipped={audio_skipped}; "
            f"effective_range={effective_min:.3f}..{effective_max:.3f}; memory_mode={memory_mode}; warnings={warning_text}"
        )
        logging.info("%s %s", DIRECTOR_PREFIX, info)
        if skipped:
            logging.info("%s skipped %d LoRA pairs; first entries: %s", DIRECTOR_PREFIX, len(skipped), skipped[:10])
        return (out, video_latent, info, envelope_data)


def _pil_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_curve(draw: ImageDraw.ImageDraw, profile: List[float], box: Tuple[int, int, int, int], color, y_min: float, y_max: float):
    if not profile:
        return
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    denom = max(1e-6, float(y_max) - float(y_min))
    pts = []
    n = len(profile)
    for i, v in enumerate(profile):
        x = x0 + (i / float(max(1, n - 1))) * width
        y = y1 - ((float(v) - float(y_min)) / denom) * height
        y = max(y0, min(y1, y))
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=3)


def _draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, y0: int, y1: int, color, *, dash: int = 7, gap: int = 5, width: int = 2):
    y = int(y0)
    x = int(round(x))
    while y < int(y1):
        draw.line((x, y, x, min(int(y1), y + dash)), fill=color, width=width)
        y += dash + gap


class LTXVTemporalSchedulePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_data": ("LTXV_SEGMENT_DATA",),
                "width": ("INT", {"default": 1200, "min": 640, "max": 2400, "step": 10}),
                "height": ("INT", {"default": 800, "min": 360, "max": 1600, "step": 10}),
                "max_text_chars": ("INT", {"default": 54, "min": 12, "max": 640, "step": 1, "tooltip": "Maximum prompt characters considered per segment. Increase preview height to make long texts visible; the available box height remains the final display limit."}),
            },
            "optional": {
                "data_1": ("LTXV_ENVELOPE_DATA",),
                "data_2": ("LTXV_ENVELOPE_DATA",),
                "data_3": ("LTXV_ENVELOPE_DATA",),
                "data_4": ("LTXV_ENVELOPE_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "preview"
    CATEGORY = "LTXV/Temporal Director"
    DESCRIPTION = "v1.3.0: seconds-first schedule inspector with segment durations, q-curve phase markers, warnings, and up to four TimeGate envelopes."

    def preview(self, segment_data: Dict, width: int, height: int, max_text_chars: int, data_1=None, data_2=None, data_3=None, data_4=None):
        data = _validate_segment_data(segment_data)
        curves = [d for d in (data_1, data_2, data_3, data_4) if isinstance(d, dict)]

        w = int(width)
        h = int(height)
        img = Image.new("RGB", (w, h), (31, 34, 40))
        draw = ImageDraw.Draw(img)
        font_title = _pil_font(max(18, int(w * 0.026)))
        font_time = _pil_font(max(17, int(w * 0.021)))
        font = _pil_font(max(12, int(w * 0.015)))
        font_small = _pil_font(max(10, int(w * 0.012)))
        margin = int(w * 0.04)

        total = int(data["total_frames"])
        fps = float(data.get("fps", 24.0))
        duration = float(data.get("duration_seconds", total / float(max(1e-6, fps))))
        time_edges = list(data.get("segment_time_edges_seconds") or [edge / float(max(1e-6, fps)) for edge in data["segment_edges"]])
        segment_durations = list(data.get("segment_durations_seconds") or [length / float(max(1e-6, fps)) for length in data["segment_lengths"]])

        legend_line_h = max(21, int((getattr(font_small, "size", 12) or 12) * 1.55))
        legend_rows = max(1, len(curves))
        chart_top = max(132, int(h * 0.245))
        chart_bottom = min(int(h * 0.72), h - (legend_rows * legend_line_h + 58))
        chart_bottom = max(chart_top + 110, chart_bottom)
        chart_bottom = min(chart_bottom, h - 70)
        chart_left = margin
        chart_right = w - margin
        chart_w = chart_right - chart_left

        draw.text((margin, 16), "LTXV Temporal Schedule", font=font_title, fill=(226, 232, 240))
        duration_text = f"{duration:.2f} s"
        draw.text((margin, 55), duration_text, font=font_time, fill=(242, 246, 252))
        try:
            duration_box = draw.textbbox((margin, 55), duration_text, font=font_time)
            details_x = int(duration_box[2] + 12)
        except Exception:
            details_x = margin + len(duration_text) * 14 + 12
        details = f"· {total} frames · {data.get('latent_frames', '?')} latent · {fps:.2f} fps · {data['segment_count']} segments"
        draw.text((details_x, 61), details, font=font, fill=(185, 193, 207))

        warnings = list(data.get("warnings") or [])
        normalizations = list(data.get("normalizations") or [])
        ok_text = "OK: prompt segments parsed"
        if normalizations:
            ok_text += " · " + "; ".join(normalizations)
        warn_text = ok_text if not warnings else "Warnings: " + "; ".join(warnings)
        draw.text((margin, 96), _truncate(warn_text, 160), font=font_small, fill=(240, 170, 90) if warnings else (135, 210, 150))

        # Background chart and grid. Seconds are the primary x-axis unit.
        draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(120, 128, 142), width=1, fill=(37, 40, 47))
        for i in range(1, 5):
            x = chart_left + i * chart_w / 5
            draw.line((x, chart_top, x, chart_bottom), fill=(67, 73, 84), width=1)
        for i in range(1, 4):
            y = chart_top + i * (chart_bottom - chart_top) / 4
            draw.line((chart_left, y, chart_right, y), fill=(67, 73, 84), width=1)

        colors = [(65, 115, 180), (195, 110, 50), (75, 145, 75), (150, 90, 180), (185, 150, 65), (75, 150, 160)]
        for i, (start, end) in enumerate(zip(data["segment_edges"][:-1], data["segment_edges"][1:])):
            x0 = chart_left + (start / float(total)) * chart_w
            x1 = chart_left + (end / float(total)) * chart_w
            color = colors[i % len(colors)]
            overlay = Image.new("RGBA", (max(1, int(x1 - x0)), chart_bottom - chart_top), color + (70,))
            img.paste(overlay, (int(x0), chart_top), overlay)
            draw.line((x0, chart_top, x0, chart_bottom), fill=(150, 157, 170), width=1)

            start_s = float(time_edges[i])
            end_s = float(time_edges[i + 1])
            label_time = f"[{i+1}] {start_s:.2f}–{end_s:.2f} s"
            label_frames = f"{end-start} frames"
            draw.text((x0 + 6, chart_top + 7), _truncate(label_time, 28), font=font_small, fill=(238, 242, 250))
            draw.text((x0 + 6, chart_top + 25), label_frames, font=font_small, fill=(194, 204, 219))

            text_raw = data["segment_texts"][i] if i < len(data["segment_texts"]) else ""
            text_for_box = _truncate(text_raw, int(max_text_chars))
            box_w = max(30, int(x1 - x0) - 12)
            available_h = max(24, (chart_bottom - chart_top) - 57)
            line_h = max(12, int((getattr(font_small, "size", 12) or 12) * 1.25))
            max_lines = max(1, available_h // line_h)
            wrapped = _wrap_text_to_width(draw, text_for_box, font_small, box_w, max_lines)
            for li, line in enumerate(wrapped):
                draw.text((x0 + 6, chart_top + 47 + li * line_h), line, font=font_small, fill=(220, 226, 238))
        draw.line((chart_right, chart_top, chart_right, chart_bottom), fill=(150, 157, 170), width=1)

        curve_colors = [(245, 245, 245), (110, 190, 255), (255, 155, 105), (185, 145, 255)]
        all_vals = []
        for c in curves:
            vals = c.get("effective_frame_profile") or c.get("frame_profile") or []
            all_vals.extend([float(x) for x in vals if isinstance(x, (int, float))])
        y_min = min(0.0, min(all_vals) if all_vals else 0.0)
        y_max = max(1.0, max(all_vals) if all_vals else 1.0)
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0

        for idx, c in enumerate(curves):
            color = curve_colors[idx % len(curve_colors)]
            vals = [float(x) for x in (c.get("effective_frame_profile") or c.get("frame_profile") or [])]
            _draw_curve(draw, vals, (chart_left, chart_top, chart_right, chart_bottom), color, y_min, y_max)

            meta = c.get("meta", {}) if isinstance(c.get("meta", {}), dict) else {}
            target = int(meta.get("target_segment", 0) or 0)
            in_shift_frames = int(meta.get("q_in_shift_frames", 0) or 0)
            out_shift_frames = int(meta.get("q_out_shift_frames", 0) or 0)
            if in_shift_frames != 0 and (target > 1 or bool(meta.get("incoming_edge_virtual", False))):
                marker_frame = int(meta.get("incoming_transition_frame", meta.get("incoming_shift_anchor_frame", meta.get("region_start", 0))))
                marker_x_raw = chart_left + (marker_frame / float(total)) * chart_w
                marker_x = max(chart_left, min(chart_right, marker_x_raw))
                marker_label = "IN END←" if marker_frame < 0 else ("IN END→" if marker_frame > total else "IN END")
                _draw_dashed_vertical(draw, marker_x, chart_top, chart_bottom, color)
                draw.text((marker_x + 3, chart_bottom - 17), marker_label, font=font_small, fill=color)
            if out_shift_frames != 0 and (target < int(data["segment_count"]) or bool(meta.get("outgoing_edge_virtual", False))):
                marker_frame = int(meta.get("outgoing_transition_frame", meta.get("outgoing_shift_anchor_frame", meta.get("region_end", total))))
                marker_x_raw = chart_left + (marker_frame / float(total)) * chart_w
                marker_x = max(chart_left, min(chart_right, marker_x_raw))
                marker_label = "OUT START←" if marker_frame < 0 else ("OUT START→" if marker_frame > total else "OUT START")
                _draw_dashed_vertical(draw, marker_x, chart_top, chart_bottom, color)
                draw.text((marker_x + 3, chart_bottom - 34), marker_label, font=font_small, fill=color)

        # Primary time-axis labels.
        tick_y = chart_bottom + 3
        for i in range(6):
            frac = i / 5.0
            x = chart_left + frac * chart_w
            label = f"{duration * frac:.2f}s"
            try:
                box = draw.textbbox((0, 0), label, font=font_small)
                tw = box[2] - box[0]
            except Exception:
                tw = len(label) * 7
            tx = x - tw / 2
            tx = max(chart_left, min(chart_right - tw, tx))
            draw.text((tx, tick_y), label, font=font_small, fill=(167, 177, 193))

        legend_top = chart_bottom + 25
        for idx, c in enumerate(curves):
            color = curve_colors[idx % len(curve_colors)]
            name = _truncate(str(c.get("lora_name") or c.get("source_node") or f"curve {idx+1}"), 38)
            meta = c.get("meta", {}) if isinstance(c.get("meta", {}), dict) else {}
            soft = int(meta.get("edge_softness_frames", 0) or 0)
            soft_seconds = soft / float(max(1e-6, fps))
            in_shift = float(meta.get("q_in_shift", 0.0) or 0.0)
            out_shift = float(meta.get("q_out_shift", 0.0) or 0.0)
            in_frames = int(meta.get("q_in_shift_frames", 0) or 0)
            out_frames = int(meta.get("q_out_shift_frames", 0) or 0)
            in_pos = int(meta.get("incoming_transition_frame", meta.get("region_start", 0)) or 0)
            out_pos = int(meta.get("outgoing_transition_frame", meta.get("region_end", total)) or 0)
            y = legend_top + idx * legend_line_h
            draw.rectangle((margin, y + 4, margin + 18, y + 14), fill=color)
            edge_tags = []
            if bool(meta.get("incoming_edge_virtual", False)):
                edge_tags.append("pre-roll")
            if bool(meta.get("outgoing_edge_virtual", False)):
                edge_tags.append("post-roll")
            edge_desc = f" · edge={'+'.join(edge_tags)}" if edge_tags else ""
            desc = (
                f"{name} · S{meta.get('target_segment', c.get('effect_region', '?'))} · strength={float(c.get('lora_strength', 0.0)):.2f} "
                f"· soft={soft_seconds:.2f}s/{soft}f · in {in_shift:+.2f} @ {in_pos / float(max(1e-6, fps)):.2f}s "
                f"· out {out_shift:+.2f} @ {out_pos / float(max(1e-6, fps)):.2f}s{edge_desc}"
            )
            draw.text((margin + 26, y), _truncate(desc, 150), font=font_small, fill=(210, 216, 228))

        report_lines = [
            "LTXV Temporal Schedule Preview v1.3.0",
            f"duration={duration:.3f}s frames={total} latent={data.get('latent_frames')} fps={fps:.3f}",
            f"segments={data['segment_count']} lengths={data['segment_lengths_csv']}",
            "segment_durations=" + ",".join(f"{float(x):.3f}s" for x in segment_durations),
            f"boundary_mode={data.get('boundary_mode', 'equal')} boundaries={','.join(f'{float(x):.4f}' for x in data.get('boundary_values_norm', [])) or 'none'}",
            "warnings=" + ("none" if not warnings else "; ".join(warnings)),
            "normalizations=" + ("none" if not normalizations else "; ".join(normalizations)),
        ]
        for i, text in enumerate(data["segment_texts"]):
            report_lines.append(
                f"segment {i+1}: time={float(time_edges[i]):.3f}-{float(time_edges[i+1]):.3f}s "
                f"frames=[{int(data['segment_edges'][i])},{int(data['segment_edges'][i+1])}) "
                f"length={int(data['segment_lengths'][i])}f; {text}"
            )
        for idx, c in enumerate(curves):
            meta = c.get("meta", {}) if isinstance(c, dict) and isinstance(c.get("meta", {}), dict) else {}
            in_frames = int(meta.get("q_in_shift_frames", 0) or 0)
            out_frames = int(meta.get("q_out_shift_frames", 0) or 0)
            in_pos = int(meta.get("incoming_transition_frame", meta.get("region_start", 0)) or 0)
            out_pos = int(meta.get("outgoing_transition_frame", meta.get("region_end", total)) or 0)
            edge_mode = []
            if bool(meta.get("incoming_edge_virtual", False)):
                edge_mode.append("virtual_pre_roll")
            if bool(meta.get("outgoing_edge_virtual", False)):
                edge_mode.append("virtual_post_roll")
            report_lines.append(
                f"curve {idx+1}: target={meta.get('target_segment', c.get('effect_region', '?') if isinstance(c, dict) else '?')} "
                f"edge_softness={int(meta.get('edge_softness_frames', 0) or 0)}f/{int(meta.get('edge_softness_frames', 0) or 0) / float(max(1e-6, fps)):.3f}s "
                f"q_in_shift={float(meta.get('q_in_shift', 0.0) or 0.0):+.2f} (delta={in_frames:+d}f/{in_frames / float(max(1e-6, fps)):+.3f}s; transition={in_pos}f/{in_pos / float(max(1e-6, fps)):.3f}s) "
                f"q_out_shift={float(meta.get('q_out_shift', 0.0) or 0.0):+.2f} (delta={out_frames:+d}f/{out_frames / float(max(1e-6, fps)):+.3f}s; transition={out_pos}f/{out_pos / float(max(1e-6, fps)):.3f}s) "
                f"edge_mode={'+'.join(edge_mode) if edge_mode else 'neighbor_segments'} "
                f"relay_epsilon={meta.get('relay_epsilon', 'none')} relay_sigma={meta.get('relay_sigma', 'none')}"
            )
        tensor = torch.from_numpy(__import__("numpy").array(img).astype("float32") / 255.0).unsqueeze(0)
        return (tensor, "\n".join(report_lines))


NODE_CLASS_MAPPINGS = {
    "LTXVScheduleSync": LTXVScheduleSync,
    "LTXVPromptRelayEncodeScheduled": LTXVPromptRelayEncodeScheduled,
    "LTXVTimeGatedLoRAScheduled": LTXVTimeGatedLoRAScheduled,
    "LTXVTemporalSchedulePreview": LTXVTemporalSchedulePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVScheduleSync": "LTXV Schedule Sync",
    "LTXVPromptRelayEncodeScheduled": "LTXV Prompt Relay Encode (Scheduled)",
    "LTXVTimeGatedLoRAScheduled": "LTXV Time-Gated LoRA (Scheduled)",
    "LTXVTemporalSchedulePreview": "LTXV Temporal Schedule Preview",
}
