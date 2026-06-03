"""
ComfyUI-LTXV-TimeGated-LoRA v1.0rc1

Production release candidate: temporal visual LoRA gating for LTX 2.3 video models.

Scope:
- one or more stacked visual temporal LoRA nodes; audio LoRA layers are deliberately ignored in the v1.0 scope
- manual frame schedules and single effect-region presets
- required video_latent input derives the actual LTX temporal length and is passed through for clean stacking
- simplified user-facing UI: required latent timing, visual LoRA only, fixed LTX 2.3 temporal compression
- stack-aware runtime nesting: multiple Time-Gated LoRA nodes may be chained before the sampler
- low-rank temporal gating reduces peak activation memory compared with v0.1.3
- LoRA deltas are applied only on Linear calls whose sequence length can be mapped
  cleanly to latent frames; non-video/token-mismatched calls are skipped to avoid
  global leakage.

Install:
  ComfyUI/custom_nodes/ComfyUI-LTXV-TimeGated-LoRA/

Node:
  LTXV Time-Gated LoRA (LTX 2.3)
"""

from __future__ import annotations

import logging
import math
import threading
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.utils


LOG_PREFIX = "[LTXV Time-Gated LoRA]"
LTX23_TEMPORAL_COMPRESSION = 8
_RUNTIME_STACK_STATE = threading.local()


def _get_lora_path(lora_name: str) -> str:
    if hasattr(folder_paths, "get_full_path_or_raise"):
        return folder_paths.get_full_path_or_raise("loras", lora_name)
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise FileNotFoundError(f"LoRA not found in ComfyUI loras folder: {lora_name}")
    return path


def _parse_csv_numbers(text: str, cast, field_name: str) -> List:
    if text is None:
        raise ValueError(f"{field_name} must not be empty")
    parts = [p.strip() for p in re.split(r"[,;\s]+", str(text).strip()) if p.strip() != ""]
    if not parts:
        raise ValueError(f"{field_name} must contain at least one value")
    out = []
    for p in parts:
        try:
            out.append(cast(p))
        except Exception as exc:
            raise ValueError(f"Could not parse value '{p}' in {field_name}") from exc
    return out


def _build_frame_profile(
    *,
    total_frames: int,
    segment_lengths: str,
    segment_strengths: str,
    transition_frames: int,
) -> torch.Tensor:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if transition_frames < 0:
        raise ValueError("transition_frames must be >= 0")

    lengths = _parse_csv_numbers(segment_lengths, int, "segment_lengths")
    strengths = _parse_csv_numbers(segment_strengths, float, "segment_strengths")

    if any(x <= 0 for x in lengths):
        raise ValueError("segment_lengths values must all be positive integers")
    if len(lengths) != len(strengths):
        raise ValueError(
            f"segment_lengths and segment_strengths must contain the same number of entries "
            f"({len(lengths)} != {len(strengths)})"
        )
    if sum(lengths) != total_frames:
        raise ValueError(
            f"sum(segment_lengths) must equal total_frames. Got sum={sum(lengths)}, "
            f"total_frames={total_frames}. In manual_frames mode the node does not auto-pad or auto-scale."
        )

    profile = torch.empty(total_frames, dtype=torch.float32)
    start = 0
    boundaries = []
    for length, strength in zip(lengths, strengths):
        end = start + length
        profile[start:end] = float(strength)
        boundaries.append((start, end))
        start = end

    # Symmetric-ish crossfade around segment boundaries. If transition_frames is 6,
    # up to 3 frames before and 3 frames after the boundary are blended.
    if transition_frames > 0 and len(strengths) > 1:
        left = transition_frames // 2
        right = transition_frames - left
        cumulative = 0
        for i in range(len(lengths) - 1):
            cumulative += lengths[i]
            prev_strength = float(strengths[i])
            next_strength = float(strengths[i + 1])
            fade_start = max(0, cumulative - left)
            fade_end = min(total_frames, cumulative + right)
            width = fade_end - fade_start
            if width <= 1:
                continue
            for frame in range(fade_start, fade_end):
                alpha = (frame - fade_start) / float(width - 1)
                profile[frame] = prev_strength * (1.0 - alpha) + next_strength * alpha

    return profile


EFFECT_REGIONS = {
    "first quarter": (0.00, 0.25),
    "second quarter": (0.25, 0.50),
    "third quarter": (0.50, 0.75),
    "last quarter": (0.75, 1.00),
    "first third": (0.00, 1.0 / 3.0),
    "middle third": (1.0 / 3.0, 2.0 / 3.0),
    "last third": (2.0 / 3.0, 1.00),
    "first half": (0.00, 0.50),
    "center half": (0.25, 0.75),
    "last half": (0.50, 1.00),
}


def _crossfade_boundary(profile: torch.Tensor, boundary: int, left_value: float, right_value: float, transition_frames: int) -> None:
    if transition_frames <= 0:
        return
    total_frames = int(profile.numel())
    left = transition_frames // 2
    right = transition_frames - left
    fade_start = max(0, int(boundary) - left)
    fade_end = min(total_frames, int(boundary) + right)
    width = fade_end - fade_start
    if width <= 1:
        return
    for frame in range(fade_start, fade_end):
        alpha = (frame - fade_start) / float(width - 1)
        profile[frame] = float(left_value) * (1.0 - alpha) + float(right_value) * alpha


def _build_effect_region_profile(
    *,
    total_frames: int,
    effect_region: str,
    strength_before: float,
    strength_during: float,
    strength_after: float,
    transition_frames: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if transition_frames < 0:
        raise ValueError("transition_frames must be >= 0")
    if effect_region not in EFFECT_REGIONS:
        raise ValueError(f"Unknown effect_region: {effect_region}")

    start_frac, end_frac = EFFECT_REGIONS[effect_region]
    region_start = int(round(total_frames * start_frac))
    region_end = int(round(total_frames * end_frac))
    region_start = max(0, min(total_frames, region_start))
    region_end = max(region_start + 1, min(total_frames, region_end))

    profile = torch.full((total_frames,), float(strength_before), dtype=torch.float32)
    profile[region_start:region_end] = float(strength_during)
    if region_end < total_frames:
        profile[region_end:] = float(strength_after)

    if region_start > 0:
        _crossfade_boundary(profile, region_start, float(strength_before), float(strength_during), int(transition_frames))
    if region_end < total_frames:
        _crossfade_boundary(profile, region_end, float(strength_during), float(strength_after), int(transition_frames))

    return profile, (region_start, region_end)


def _infer_ltx_latent_frames(video_latent) -> Tuple[int, str]:
    if video_latent is None:
        raise ValueError("video_latent is not connected")
    samples = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
    if not torch.is_tensor(samples):
        raise ValueError("video_latent does not contain a tensor under 'samples'")
    if samples.ndim != 5:
        raise ValueError(
            f"video_latent must be a video-only LATENT with samples shaped [B,C,T,H,W]. "
            f"Got shape={tuple(samples.shape)}. Connect the video latent before any audio/video concat node."
        )
    latent_frames = int(samples.shape[2])
    if latent_frames <= 0:
        raise ValueError(f"video_latent has invalid temporal dimension: shape={tuple(samples.shape)}")
    return latent_frames, f"video_latent shape={tuple(samples.shape)}"


def _resolve_timing(video_latent) -> Tuple[int, int, str]:
    """Resolve the timeline from the final video-only latent supplied by the workflow."""
    if video_latent is None:
        raise ValueError(
            "video_latent is required. Connect the final video-only LATENT after image/FLF conditioning "
            "and before any audio/video concatenation node."
        )
    latent_frames, source = _infer_ltx_latent_frames(video_latent)
    resolved_frames = (latent_frames - 1) * LTX23_TEMPORAL_COMPRESSION + 1
    return resolved_frames, latent_frames, source


def _frame_profile_to_latent_profile(
    frame_profile: torch.Tensor,
    *,
    temporal_compression: int,
) -> torch.Tensor:
    if temporal_compression <= 0:
        raise ValueError("temporal_compression must be > 0")
    total_frames = int(frame_profile.numel())
    latent_frames = ((total_frames - 1) // temporal_compression) + 1
    values = []
    for i in range(latent_frames):
        start = i * temporal_compression
        end = min(total_frames, start + temporal_compression)
        if start >= total_frames:
            values.append(frame_profile[-1])
        else:
            values.append(frame_profile[start:end].mean())
    return torch.stack(values).to(torch.float32)


@dataclass(frozen=True)
class LoRAPair:
    down_key: str
    up_key: str
    alpha_key: Optional[str]
    module_hint: str


@dataclass
class RuntimeLoRASpec:
    module_name: str
    down: torch.Tensor
    up: torch.Tensor
    scale: float
    strength: float
    latent_profile: torch.Tensor
    cached_base: Optional[nn.Module] = None
    cached_wrapper: Optional[nn.Module] = None


def _strip_lora_suffix(key: str) -> Optional[Tuple[str, str]]:
    """Return (module_hint, part) for common LoRA tensor key styles."""
    suffixes = {
        ".lora_down.weight": "down",
        ".lora_up.weight": "up",
        ".lora_A.weight": "down",
        ".lora_B.weight": "up",
        ".lora_down.weight_patched": "down",
        ".lora_up.weight_patched": "up",
    }
    for suffix, part in suffixes.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    return None


def _find_lora_pairs(sd: Dict[str, torch.Tensor]) -> List[LoRAPair]:
    grouped: Dict[str, Dict[str, str]] = {}
    for key in sd.keys():
        stripped = _strip_lora_suffix(key)
        if stripped is None:
            continue
        hint, part = stripped
        grouped.setdefault(hint, {})[part] = key

    pairs: List[LoRAPair] = []
    for hint, parts in grouped.items():
        if "down" not in parts or "up" not in parts:
            continue
        alpha_key = None
        for candidate in (hint + ".alpha", hint + ".lora_alpha"):
            if candidate in sd:
                alpha_key = candidate
                break
        pairs.append(LoRAPair(parts["down"], parts["up"], alpha_key, hint))
    return pairs


def _normalize_hint_candidates(hint: str) -> List[str]:
    """Generate plausible module names from common LoRA naming conventions.

    LTX LoRAs in the wild use at least two naming families:

    1. Comfy-style / native style, for example
       diffusion_model.transformer_blocks.0.attn1.to_q

    2. Diffusers/Lightricks-style, for example
       diffusion_model.blocks.0.self_attn.q

    v0.1.3 adds conservative aliases from the second form to the first.
    """
    candidates = []

    def add(x: str):
        x = x.strip(".")
        if x and x not in candidates:
            candidates.append(x)

    raw = hint
    add(raw)

    prefixes_to_strip = [
        "model.",
        "diffusion_model.",
        "unet.",
        "transformer.",
        "lora_unet_",
        "lora_te_",
    ]
    for p in prefixes_to_strip:
        if raw.startswith(p):
            add(raw[len(p) :])

    # Kohya-style flattened module paths often use underscores instead of dots.
    # Keep this conservative: try only after known LoRA prefixes and as fallback.
    flattened = raw
    for p in ("lora_unet_", "lora_te_"):
        if flattened.startswith(p):
            flattened = flattened[len(p) :]
    add(flattened.replace("_", "."))

    # Additional common wrappers.
    for c in list(candidates):
        if c.startswith("diffusion.model."):
            add(c[len("diffusion.model.") :])
        if c.startswith("model.diffusion_model."):
            add(c[len("model.") :])
            add(c[len("model.diffusion_model.") :])

    # LTX diffusers/Lightricks LoRA key aliases -> Comfy/LTXVideo module names.
    # Attention mapping:
    #   blocks.N.self_attn.q/k/v/o  -> transformer_blocks.N.attn1.to_q/to_k/to_v/to_out.0
    #   blocks.N.cross_attn.q/k/v/o -> transformer_blocks.N.attn2.to_q/to_k/to_v/to_out.0
    # We create variants with and without diffusion_model. because different wrappers
    # expose named_modules() differently.
    def add_ltx_aliases(c: str):
        m = re.match(r"^(?:diffusion_model\.)?blocks\.(\d+)\.(self_attn|cross_attn)\.(q|k|v|o)$", c)
        if m:
            idx, attn_kind, proj = m.groups()
            attn = "attn1" if attn_kind == "self_attn" else "attn2"
            target = "to_out.0" if proj == "o" else f"to_{proj}"
            base = f"transformer_blocks.{idx}.{attn}.{target}"
            add(base)
            add("diffusion_model." + base)

        # Feed-forward aliases. These are intentionally broad because LTX
        # checkpoints/nodes expose FFN modules under slightly different names.
        m = re.match(r"^(?:diffusion_model\.)?blocks\.(\d+)\.ffn\.(0|2)$", c)
        if m:
            idx, part = m.groups()
            bases = [
                f"transformer_blocks.{idx}.ff.net.{part}",
                f"transformer_blocks.{idx}.ff.net.{part}.proj",
                f"transformer_blocks.{idx}.ffn.{part}",
                f"transformer_blocks.{idx}.feed_forward.net.{part}",
                f"transformer_blocks.{idx}.feed_forward.net.{part}.proj",
            ]
            for base in bases:
                add(base)
                add("diffusion_model." + base)

    for c in list(candidates):
        add_ltx_aliases(c)

    return candidates


def _resolve_module_name(module_hint: str, modules: Dict[str, nn.Module]) -> Optional[str]:
    candidates = _normalize_hint_candidates(module_hint)
    for c in candidates:
        if c in modules:
            return c

    # Suffix match as a last resort. Only accept a unique match.
    for c in candidates:
        matches = [name for name in modules.keys() if name.endswith("." + c) or name == c]
        if len(matches) == 1:
            return matches[0]
    return None


class TemporalLoRALinearWrapper(nn.Module):
    """A transient layer wrapper used only while one model forward is executing.

    v0.1.3 must never be installed persistently via ModelPatcher.add_object_patch().
    A persistent module replacement contaminates the shared Comfy model object across
    queue executions. The runtime model wrapper below inserts this module temporarily
    and restores the original in a finally block.
    """
    _ltx23_time_gated_wrapper = True

    def __init__(
        self,
        base: nn.Module,
        down: torch.Tensor,
        up: torch.Tensor,
        scale: float,
        strength: float,
        latent_profile: torch.Tensor,
        *,
        name: str,
        non_video_token_mode: str = "skip",
        memory_mode: str = "optimized_safe",
    ):
        super().__init__()
        self.base = base
        self.lora_down_cpu = down.detach().clone().to(device="cpu", dtype=torch.float32)
        self.lora_up_cpu = up.detach().clone().to(device="cpu", dtype=torch.float32)
        self.scale = float(scale)
        self.strength = float(strength)
        self.latent_profile_cpu = latent_profile.detach().clone().to(device="cpu", dtype=torch.float32)
        self.name = name
        self.non_video_token_mode = non_video_token_mode
        self.memory_mode = memory_mode
        self._warned_unmappable = False
        self._cache_key = None
        self._down_runtime = None
        self._up_runtime = None
        self._profile_runtime = None

    def clear_cache(self):
        self._cache_key = None
        self._down_runtime = None
        self._up_runtime = None
        self._profile_runtime = None

    def _runtime_tensors(self, x: torch.Tensor):
        key = (x.device.type, x.device.index, x.dtype)
        if key != self._cache_key:
            self._down_runtime = self.lora_down_cpu.to(device=x.device, dtype=x.dtype)
            self._up_runtime = self.lora_up_cpu.to(device=x.device, dtype=x.dtype)
            self._profile_runtime = self.latent_profile_cpu.to(device=x.device, dtype=x.dtype)
            self._cache_key = key
        return self._down_runtime, self._up_runtime, self._profile_runtime

    def _gate_for(self, x: torch.Tensor, profile: torch.Tensor) -> Optional[torch.Tensor]:
        latent_frames = int(profile.numel())
        if latent_frames <= 0 or x.ndim < 3:
            return None

        seq_len = int(x.shape[-2])
        if seq_len <= 0 or seq_len % latent_frames != 0:
            if not self._warned_unmappable:
                logging.debug(
                    "%s %s: skipping non-video or unmappable token sequence len=%s, latent_frames=%s",
                    LOG_PREFIX, self.name, seq_len, latent_frames,
                )
                self._warned_unmappable = True
            if self.non_video_token_mode == "mean":
                return profile.mean()
            return None

        tokens_per_latent_frame = seq_len // latent_frames
        gate = profile.repeat_interleave(tokens_per_latent_frame)
        view_shape = [1] * x.ndim
        view_shape[-2] = seq_len
        view_shape[-1] = 1
        return gate.view(*view_shape)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        y = self.base(x, *args, **kwargs)
        down, up, profile = self._runtime_tensors(x)
        gate = self._gate_for(x, profile)
        if gate is None:
            return y

        # v0.6 VRAM path: apply the temporal gate while the activation is still
        # low-rank. v0.1.3 multiplied a full-width [.., out_features] delta by
        # the gate, temporarily materialising another large activation tensor.
        low_rank = F.linear(x, down)
        low_rank = low_rank * gate
        delta = F.linear(low_rank, up)
        coefficient = self.scale * self.strength
        if self.memory_mode == "optimized_low_vram_inplace":
            return y.add_(delta, alpha=coefficient)
        return torch.add(y, delta, alpha=coefficient)

def _linear_dims(module: nn.Module) -> Optional[Tuple[int, int]]:
    """Return (in_features, out_features) for torch/comfy Linear-like modules."""
    # Tolerate inspection of a stale wrapper from a pre-v0.1.3 run long enough
    # to emit a useful restart error rather than a misleading shape error.
    if type(module).__name__ == "TemporalLoRALinearWrapper" and hasattr(module, "base"):
        module = module.base
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is not None and out_features is not None:
        try:
            return int(in_features), int(out_features)
        except Exception:
            pass

    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight) and weight.ndim == 2:
        # Linear weight convention: [out_features, in_features]
        return int(weight.shape[1]), int(weight.shape[0])

    return None


def _is_linear_lora_pair(down: torch.Tensor, up: torch.Tensor, module: nn.Module) -> bool:
    if down.ndim != 2 or up.ndim != 2:
        return False
    dims = _linear_dims(module)
    if dims is None:
        return False
    module_in, module_out = dims
    rank_down, in_features = int(down.shape[0]), int(down.shape[1])
    out_features, rank_up = int(up.shape[0]), int(up.shape[1])
    return (
        rank_down == rank_up
        and in_features == module_in
        and out_features == module_out
    )


def _module_debug_shape(module: nn.Module) -> str:
    dims = _linear_dims(module)
    if dims is None:
        return "module_dims=unknown"
    module_in, module_out = dims
    return f"module_in={module_in} module_out={module_out}"


def _alpha_from_sd(sd: Dict[str, torch.Tensor], pair: LoRAPair, rank: int) -> float:
    if pair.alpha_key is None:
        return float(rank)
    alpha = sd[pair.alpha_key]
    if torch.is_tensor(alpha):
        return float(alpha.detach().cpu().flatten()[0].item())
    return float(alpha)


def _get_nested_attr(root, path: str):
    obj = root
    for part in path.split("."):
        if part.isdigit() and isinstance(obj, (list, tuple, nn.ModuleList, nn.Sequential)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested_attr(root, path: str, value):
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(obj, (list, tuple, nn.ModuleList, nn.Sequential)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if last.isdigit() and isinstance(obj, (list, nn.ModuleList, nn.Sequential)):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


class TemporalLoRAModelFunctionWrapper:
    """Installs temporal LoRA wrappers only for the duration of each model call.

    v0.7 supports multiple chained Time-Gated LoRA nodes.  During sampling, the
    outer node temporarily installs its linear wrappers and then calls the prior
    model-function wrapper; an inner Time-Gated node therefore legitimately sees
    transient TemporalLoRALinearWrapper bases and wraps them additively.
    """
    def __init__(self, model_root, specs: List[RuntimeLoRASpec], previous_wrapper=None, memory_mode: str = "optimized_safe"):
        self.model_root = model_root
        self.specs = specs
        self.previous_wrapper = previous_wrapper
        self.memory_mode = memory_mode
        self.lock = threading.RLock()

    def cleanup(self, **kwargs):
        for spec in self.specs:
            if spec.cached_wrapper is not None and hasattr(spec.cached_wrapper, "clear_cache"):
                spec.cached_wrapper.clear_cache()

    def __call__(self, model_function, args):
        originals = []
        parent_depth = int(getattr(_RUNTIME_STACK_STATE, "depth", 0))
        with self.lock:
            _RUNTIME_STACK_STATE.depth = parent_depth + 1
            try:
                for spec in self.specs:
                    current = _get_nested_attr(self.model_root, spec.module_name)
                    is_temporal_wrapper = bool(getattr(current, "_ltx23_time_gated_wrapper", False))
                    if is_temporal_wrapper and parent_depth == 0:
                        raise RuntimeError(
                            "Persistent TemporalLoRALinearWrapper detected before temporal-node nesting began. "
                            "This live model may have been contaminated by an older pre-v0.1.3 build. "
                            "Restart ComfyUI once before testing v0.7."
                        )
                    # If parent_depth > 0, current is a transient wrapper installed by another
                    # chained Time-Gated node during this same forward call. Wrapping it is
                    # intentional: base(x) carries the prior LoRA delta and this wrapper adds
                    # its own scheduled delta.
                    if spec.cached_wrapper is None or spec.cached_base is not current:
                        spec.cached_base = current
                        spec.cached_wrapper = TemporalLoRALinearWrapper(
                            base=current,
                            down=spec.down,
                            up=spec.up,
                            scale=spec.scale,
                            strength=spec.strength,
                            latent_profile=spec.latent_profile,
                            name=spec.module_name,
                            non_video_token_mode="skip",
                            memory_mode=self.memory_mode,
                        )
                    originals.append((spec.module_name, current))
                    _set_nested_attr(self.model_root, spec.module_name, spec.cached_wrapper)

                if self.previous_wrapper is not None:
                    return self.previous_wrapper(model_function, args)
                return model_function(args["input"], args["timestep"], **args["c"])
            finally:
                for name, original in reversed(originals):
                    _set_nested_attr(self.model_root, name, original)
                _RUNTIME_STACK_STATE.depth = parent_depth


class ApplyTimeGatedLoRAToModelLTX23:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Completed LTX 2.3 MODEL chain. Place this node last, immediately before the sampler/guider. Chain additional LTXV Time-Gated LoRA nodes here to combine scheduled visual effects."
                }),
                "video_latent": ("LATENT", {
                    "tooltip": "REQUIRED timing reference: connect the final video-only LATENT after image/FLF conditioning and before AV concatenation. Do not connect an audio+video latent. The unchanged latent is passed through on the output for stacking additional LTXV Time-Gated LoRA nodes."
                }),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Visual LTX 2.3 LoRA to time-gate. Audio LoRA layers are intentionally ignored in the v1.0 scope."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Global LoRA multiplier. Effective strength equals strength_model × the active schedule value. LTX 2.3 LoRAs may require strengths above 1.0; signed values are valid for slider/transform LoRAs."
                }),
                "schedule_mode": (["effect_region", "manual_frames"], {
                    "default": "effect_region",
                    "tooltip": "Choose effect_region for simple placement or manual_frames for exact segment control. REQUIRED: connect the final video-only video_latent so the schedule follows the actual rendered LTX timeline. With PromptRelay, leave segment_lengths empty for equal thirds/quarters/halves and use matching | prompt segments."
                }),
                "effect_region": (list(EFFECT_REGIONS.keys()), {
                    "default": "middle third",
                    "tooltip": "Used only in effect_region mode. Select the interval controlled by strength_during."
                }),
                "strength_before": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Used only in effect_region mode. Schedule multiplier before the selected region."
                }),
                "strength_during": ("FLOAT", {
                    "default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Used only in effect_region mode. Schedule multiplier inside the selected region."
                }),
                "strength_after": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Used only in effect_region mode. Schedule multiplier after the selected region."
                }),
                "segment_lengths": ("STRING", {
                    "default": "80,81,80", "multiline": False,
                    "tooltip": "Used only in manual_frames mode. Comma-separated segment lengths in rendered frames; the sum must equal the video length resolved from video_latent."
                }),
                "segment_strengths": ("STRING", {
                    "default": "0,1,0", "multiline": False,
                    "tooltip": "Used only in manual_frames mode. Comma-separated multipliers paired with segment_lengths; signed and decimal values are valid, for example 0,-1,1.5."
                }),
                "transition_frames": ("INT", {
                    "default": 16, "min": 0, "max": 2048, "step": 1,
                    "tooltip": "Boundary crossfade in rendered frames. Best practice: start with 8–16 frames for ordinary style/effect reveals (about 0.33–0.67 s at 24 fps); use 0 only for intentionally hard switches, and try 16–32 for strong transformations."
                }),
                "memory_mode": (["optimized_safe", "optimized_low_vram_inplace"], {
                    "default": "optimized_safe",
                    "tooltip": "optimized_safe is the recommended default. optimized_low_vram_inplace may reduce peak VRAM for long clips or stacked nodes but is experimental; use it only after an OOM in safe mode and compare output stability."
                }),
            },
        }


    RETURN_TYPES = ("MODEL", "LATENT", "STRING")
    RETURN_NAMES = ("model", "video_latent", "schedule_info")
    FUNCTION = "apply"
    CATEGORY = "LTXV/LoRA"
    DESCRIPTION = (
        "v1.0rc1: stackable runtime-safe visual time-gated LoRA for LTX 2.3. "
        "Connect the final video-only latent for timeline resolution; it is passed through for clean multi-node stacking. "
        "Chain one or more instances last in the MODEL path before the sampler/guider. Audio layers are intentionally ignored."
    )

    def apply(
        self,
        model,
        video_latent,
        lora_name: str,
        strength_model: float,
        schedule_mode: str,
        effect_region: str,
        strength_before: float,
        strength_during: float,
        strength_after: float,
        segment_lengths: str,
        segment_strengths: str,
        transition_frames: int,
        memory_mode: str,
    ):
        temporal_compression = LTX23_TEMPORAL_COMPRESSION
        resolved_frames, resolved_latent_frames, timing_source = _resolve_timing(video_latent)
        audio_note = "audio_layers=ignored_by_design"

        if schedule_mode == "manual_frames":
            frame_profile = _build_frame_profile(
                total_frames=resolved_frames,
                segment_lengths=segment_lengths,
                segment_strengths=segment_strengths,
                transition_frames=int(transition_frames),
            )
            schedule_description = f"manual_frames lengths={segment_lengths} strengths={segment_strengths}"
        elif schedule_mode == "effect_region":
            frame_profile, (region_start, region_end) = _build_effect_region_profile(
                total_frames=resolved_frames,
                effect_region=effect_region,
                strength_before=float(strength_before),
                strength_during=float(strength_during),
                strength_after=float(strength_after),
                transition_frames=int(transition_frames),
            )
            schedule_description = (
                f"effect_region='{effect_region}' frames=[{region_start},{region_end}) "
                f"strengths={float(strength_before):.3f},{float(strength_during):.3f},{float(strength_after):.3f}"
            )
        else:
            raise ValueError(f"Unsupported schedule_mode: {schedule_mode}")

        latent_profile = _frame_profile_to_latent_profile(
            frame_profile,
            temporal_compression=temporal_compression,
        )
        if int(latent_profile.numel()) != int(resolved_latent_frames):
            raise RuntimeError(
                f"Internal timing mismatch: profile latent_frames={int(latent_profile.numel())}, "
                f"resolved_latent_frames={resolved_latent_frames}."
            )

        if abs(float(strength_model)) < 1e-8 or float(frame_profile.abs().max()) < 1e-8:
            return (
                model,
                video_latent,
                f"v1.0rc1 passthrough; {schedule_description}; frames={resolved_frames}; "
                f"latent_frames={resolved_latent_frames}; transition_frames={transition_frames}; {audio_note}",
            )

        lora_path = _get_lora_path(lora_name)
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        pairs = _find_lora_pairs(sd)
        if not pairs:
            raise ValueError(
                f"No supported LoRA tensors found in {lora_name}. v1.0rc1 supports keys ending in "
                f".lora_down.weight/.lora_up.weight or .lora_A.weight/.lora_B.weight."
            )

        out = model.clone()
        modules = dict(out.model.named_modules())
        stale_wrappers = [n for n, m in modules.items() if type(m).__name__ == "TemporalLoRALinearWrapper"]
        if stale_wrappers:
            raise RuntimeError(
                "Persistent TemporalLoRALinearWrapper objects from v0.1-v0.1.2 are already present "
                "in the live model. Restart ComfyUI before testing v1.0rc1. "
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
            # Audio is expressly out of scope for v1.0.
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
            raise ValueError(
                f"No visual LoRA layers could be applied to this MODEL. First skipped entries: {details}"
            )

        previous_wrapper = out.model_options.get("model_function_wrapper")
        runtime_wrapper = TemporalLoRAModelFunctionWrapper(
            out.model, specs, previous_wrapper, memory_mode=memory_mode
        )
        out.set_model_unet_function_wrapper(runtime_wrapper)

        info = (
            f"v1.0rc1; lora={lora_name}; mode={schedule_mode}; {schedule_description}; "
            f"frames={resolved_frames}; latent_frames={resolved_latent_frames}; transition_frames={transition_frames}; "
            f"prepared_visual_layers={len(specs)}; audio_skipped={audio_skipped}; "
            f"effective_range={float(strength_model) * float(frame_profile.min()):.3f}..{float(strength_model) * float(frame_profile.max()):.3f}; "
            f"memory_mode={memory_mode}; {audio_note}; place_last_in_MODEL_chain_before_sampler_or_guider"
        )
        logging.info("%s %s", LOG_PREFIX, info)
        if skipped:
            logging.info("%s skipped first entries: %s", LOG_PREFIX, skipped[:10])
        return (out, video_latent, info)


NODE_CLASS_MAPPINGS = {
    "ApplyTimeGatedLoRAToModelLTX23": ApplyTimeGatedLoRAToModelLTX23,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keep the internal node key stable so v0.5 workflows still resolve the class.
    "ApplyTimeGatedLoRAToModelLTX23": "LTXV Time-Gated LoRA (LTX 2.3)",
}
