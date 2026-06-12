"""
LTXV Temporal Envelope Inspector v1.1

Development/audit node for ComfyUI-LTXV-TimeGated-LoRA.

This node does NOT patch the model and does NOT load a LoRA. It only resolves the
video timeline from a video-only LTX LATENT and visualizes/reports the temporal
strength envelope that would be used for time-gated LoRA work.

Primary goals:
- inspect transition_frames boundary-crossfade behavior
- prototype the planned v1.1 local/hold_strength Q-envelope semantics
- expose warnings before long artistic transitions are used in real sampling
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import torch


LTX23_TEMPORAL_COMPRESSION = 8


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


ENVELOPE_MODES = [
    "transition_frames",
    "v1.1_local",
    "v1.1_hold_strength",
]


RAMP_MODES = [
    "flat",
    "q_curve",
]


def _infer_ltx_latent_frames(video_latent) -> Tuple[int, str]:
    if video_latent is None:
        raise ValueError("video_latent is not connected")
    samples = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
    if not torch.is_tensor(samples):
        raise ValueError("video_latent does not contain a tensor under 'samples'")
    if samples.ndim != 5:
        raise ValueError(
            "video_latent must be a video-only LATENT with samples shaped [B,C,T,H,W]. "
            f"Got shape={tuple(samples.shape)}. Connect the video latent before any audio/video concat node."
        )
    latent_frames = int(samples.shape[2])
    if latent_frames <= 0:
        raise ValueError(f"video_latent has invalid temporal dimension: shape={tuple(samples.shape)}")
    return latent_frames, f"video_latent shape={tuple(samples.shape)}"


def _resolve_timing(video_latent) -> Tuple[int, int, str]:
    latent_frames, source = _infer_ltx_latent_frames(video_latent)
    resolved_frames = (latent_frames - 1) * LTX23_TEMPORAL_COMPRESSION + 1
    return resolved_frames, latent_frames, source


def _region_to_frames(total_frames: int, effect_region: str) -> Tuple[int, int]:
    if effect_region not in EFFECT_REGIONS:
        raise ValueError(f"Unknown effect_region: {effect_region}")
    start_frac, end_frac = EFFECT_REGIONS[effect_region]
    region_start = int(round(total_frames * start_frac))
    region_end = int(round(total_frames * end_frac))
    region_start = max(0, min(total_frames, region_start))
    region_end = max(region_start + 1, min(total_frames, region_end))
    return region_start, region_end


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


def _build_v10_profile(
    *,
    total_frames: int,
    effect_region: str,
    strength_before: float,
    strength_during: float,
    strength_after: float,
    transition_frames: int,
) -> Tuple[torch.Tensor, Dict]:
    region_start, region_end = _region_to_frames(total_frames, effect_region)
    profile = torch.full((total_frames,), float(strength_before), dtype=torch.float32)
    profile[region_start:region_end] = float(strength_during)
    if region_end < total_frames:
        profile[region_end:] = float(strength_after)

    if region_start > 0:
        _crossfade_boundary(profile, region_start, float(strength_before), float(strength_during), int(transition_frames))
    if region_end < total_frames:
        _crossfade_boundary(profile, region_end, float(strength_during), float(strength_after), int(transition_frames))

    left = int(transition_frames) // 2
    right = int(transition_frames) - left
    meta = {
        "region_start": region_start,
        "region_end": region_end,
        "before_neighbor_start": 0,
        "after_neighbor_end": total_frames,
        "transition_split_left": left,
        "transition_split_right": right,
        "entry_fade": [max(0, region_start - left), min(total_frames, region_start + right)] if region_start > 0 and transition_frames > 0 else None,
        "exit_fade": [max(0, region_end - left), min(total_frames, region_end + right)] if region_end < total_frames and transition_frames > 0 else None,
    }
    return profile, meta


def _q_mix(t: torch.Tensor, q: float) -> torch.Tensor:
    """Front-loaded power curve.

    This intentionally follows the roadmap semantics chosen by the user:
    q=1 is linear, q>1 changes steeply at the start of the segment, q<1 delays
    the change toward the end of the segment.  Mathematically this is the
    ease-out family 1 - (1 - t)^q, not t^q.
    """
    q = max(float(q), 1e-6)
    return 1.0 - torch.pow(1.0 - torch.clamp(t, 0.0, 1.0), q)


def _fill_ramp(profile: torch.Tensor, start: int, end: int, value_a: float, value_b: float, ramp_mode: str, ramp_q: float) -> None:
    total_frames = int(profile.numel())
    start = max(0, min(total_frames, int(start)))
    end = max(start, min(total_frames, int(end)))
    width = end - start
    if width <= 0:
        return

    if ramp_mode == "flat":
        profile[start:end] = float(value_a)
        return

    # q_curve: include both conceptual endpoints as far as discrete frames allow.
    if width == 1:
        profile[start:end] = float(value_b)
        return
    t = torch.linspace(0.0, 1.0, steps=width, dtype=torch.float32)
    mix = _q_mix(t, ramp_q)
    profile[start:end] = float(value_a) + (float(value_b) - float(value_a)) * mix


def _build_v11_profile(
    *,
    total_frames: int,
    effect_region: str,
    strength_before: float,
    strength_during: float,
    strength_after: float,
    outside_strength: float,
    ramp_mode: str,
    ramp_q: float,
    hold_strength: bool,
) -> Tuple[torch.Tensor, Dict]:
    if ramp_mode not in RAMP_MODES:
        raise ValueError(f"Unknown ramp_mode: {ramp_mode}")
    region_start, region_end = _region_to_frames(total_frames, effect_region)
    region_len = max(1, region_end - region_start)

    before_start = max(0, region_start - region_len)
    after_end = min(total_frames, region_end + region_len)

    # Snap one-frame rounding leftovers at the clip boundaries.
    # Example: 281 rendered frames split into thirds resolves the middle third as
    # [94, 187), length 93.  A naive same-length neighbor would be [1, 94)
    # and [187, 280), leaving frame 0 and frame 280 as outside_strength.
    # Conceptually, for edge-adjacent neighbor segments, those one-frame residuals
    # belong to the neighboring segment caused by integer rounding, not to outside.
    if before_start <= 1:
        before_start = 0
    if total_frames - after_end <= 1:
        after_end = total_frames

    # v1.1rc7 semantics:
    # local and hold_strength both hold strength_before/after outside the immediate
    # ramp segments. outside_strength is kept only as legacy/diagnostic metadata and
    # no longer overrides the later/earlier timeline in local mode.
    profile = torch.empty((total_frames,), dtype=torch.float32)
    profile[:region_start] = float(strength_before)
    profile[region_start:region_end] = float(strength_during)
    if region_end < total_frames:
        profile[region_end:] = float(strength_after)

    if ramp_mode == "q_curve":
        if before_start < region_start:
            _fill_ramp(profile, before_start, region_start, float(strength_before), float(strength_during), ramp_mode, ramp_q)
        if region_end < after_end:
            _fill_ramp(profile, region_end, after_end, float(strength_during), float(strength_after), ramp_mode, ramp_q)

    meta = {
        "region_start": region_start,
        "region_end": region_end,
        "region_len": region_len,
        "before_neighbor_start": before_start,
        "before_neighbor_end": region_start,
        "after_neighbor_start": region_end,
        "after_neighbor_end": after_end,
        "hold_strength": bool(hold_strength),
        "outside_strength": None,
        "q_formula": "mix = 1 - (1 - t) ** q; q>1 front-loaded/steeper at segment start, q<1 delayed/later change",
    }
    return profile, meta


def _frame_profile_to_latent_profile(frame_profile: torch.Tensor, temporal_compression: int) -> torch.Tensor:
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


def _profile_samples(profile: torch.Tensor) -> List[Dict]:
    total = int(profile.numel())
    if total <= 0:
        return []
    points = [0, 0.125, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 0.875, 1.0]
    rows = []
    seen = set()
    for p in points:
        idx = int(round((total - 1) * p))
        if idx in seen:
            continue
        seen.add(idx)
        rows.append({"frame": idx, "pos": round(idx / max(1, total - 1), 4), "value": round(float(profile[idx]), 6)})
    return rows


def _plateau_frames(profile: torch.Tensor, target: float, eps: float = 1e-5) -> int:
    return int((torch.abs(profile - float(target)) <= eps).sum().item())


def _range_text(start: int, end: int) -> str:
    if end <= start:
        return "empty"
    return f"[{start}, {end}) len={end - start}"


def _build_warnings(
    *,
    envelope_mode: str,
    ramp_mode: str,
    ramp_q: float,
    transition_frames: int,
    meta: Dict,
    total_frames: int,
    strength_before: float,
    strength_during: float,
    strength_after: float,
) -> List[str]:
    warnings: List[str] = []
    if envelope_mode not in ("transition_frames", "v1.0rc1_transition_frames") and int(transition_frames) != 0:
        warnings.append("transition_frames is ignored in v1.1 envelope modes; segment length defines the ramp duration.")
    if ramp_mode == "flat":
        warnings.append("ramp_q is ignored in flat mode.")
    if ramp_mode == "q_curve" and abs(float(ramp_q) - 1.0) < 1e-6:
        warnings.append("q_curve with q=1.0 is linear.")
    if envelope_mode in ("transition_frames", "v1.0rc1_transition_frames"):
        entry = meta.get("entry_fade")
        exit_ = meta.get("exit_fade")
        if entry and exit_:
            overlap = min(entry[1], exit_[1]) - max(entry[0], exit_[0])
            if overlap > 0:
                warnings.append(f"v1.0 transition crossfades overlap by {overlap} rendered frames; this can overwrite part of the earlier boundary blend.")
            plateau_est = max(0, int(meta["region_end"]) - int(meta["region_start"]) - int(transition_frames))
            if plateau_est == 0 and abs(float(strength_during) - float(strength_before)) > 1e-8 and abs(float(strength_during) - float(strength_after)) > 1e-8:
                warnings.append("The full-strength active plateau may be eliminated by the transition width.")
    else:
        if meta.get("before_neighbor_end", 0) <= meta.get("before_neighbor_start", 0):
            warnings.append("No preceding neighbor segment exists for this region; strength_before has no local ramp area.")
        if meta.get("after_neighbor_end", 0) <= meta.get("after_neighbor_start", 0):
            warnings.append("No following neighbor segment exists for this region; strength_after has no local ramp area.")
        if ramp_mode == "q_curve":
            if abs(float(strength_before) - float(strength_during)) < 1e-8:
                warnings.append("strength_before equals strength_during; incoming q-curve is visually flat.")
            if abs(float(strength_after) - float(strength_during)) < 1e-8:
                warnings.append("strength_after equals strength_during; outgoing q-curve is visually flat.")
    return warnings


def _make_plot_image(profile: torch.Tensor, latent_profile: torch.Tensor, width: int = 1024, height: int = 256) -> torch.Tensor:
    # Simple tensor-only plot image: white background, light grid, dark frame-profile curve,
    # and a secondary latent-profile curve.  Output shape: [1,H,W,3], Comfy IMAGE.
    img = torch.ones((height, width, 3), dtype=torch.float32)
    # grid
    for x in range(0, width, max(1, width // 8)):
        img[:, x : min(width, x + 1), :] = 0.88
    for y in range(0, height, max(1, height // 4)):
        img[y : min(height, y + 1), :, :] = 0.88

    values = profile.detach().cpu().to(torch.float32)
    if values.numel() == 0:
        return img.unsqueeze(0)
    vmin = float(torch.min(values).item())
    vmax = float(torch.max(values).item())
    if abs(vmax - vmin) < 1e-8:
        vmin -= 0.5
        vmax += 0.5

    def y_from_value(v: float) -> int:
        norm = (float(v) - vmin) / max(1e-8, (vmax - vmin))
        y = int(round((height - 12) - norm * (height - 24)))
        return max(0, min(height - 1, y))

    # zero line if visible
    if vmin <= 0.0 <= vmax:
        yz = y_from_value(0.0)
        img[max(0, yz - 1) : min(height, yz + 1), :, :] = 0.70

    # draw frame profile in black-ish
    n = int(values.numel())
    prev_x = 0
    prev_y = y_from_value(float(values[0]))
    for xi in range(width):
        idx = int(round(xi * (n - 1) / max(1, width - 1)))
        yi = y_from_value(float(values[idx]))
        x0, x1 = sorted((prev_x, xi))
        y0, y1 = sorted((prev_y, yi))
        if x1 == x0:
            img[y0 : y1 + 1, xi, :] = 0.05
        else:
            for xx in range(x0, x1 + 1):
                t = (xx - x0) / max(1, x1 - x0)
                yy = int(round(prev_y * (1.0 - t) + yi * t))
                img[max(0, yy - 1) : min(height, yy + 2), xx, :] = 0.05
        prev_x, prev_y = xi, yi

    # draw latent profile as medium gray dots/line
    lat = latent_profile.detach().cpu().to(torch.float32)
    if lat.numel() > 1:
        prev_x = 0
        prev_y = y_from_value(float(lat[0]))
        for i in range(int(lat.numel())):
            xi = int(round(i * (width - 1) / max(1, int(lat.numel()) - 1)))
            yi = y_from_value(float(lat[i]))
            x0, x1 = sorted((prev_x, xi))
            for xx in range(x0, x1 + 1):
                t = (xx - x0) / max(1, x1 - x0)
                yy = int(round(prev_y * (1.0 - t) + yi * t))
                img[max(0, yy - 1) : min(height, yy + 2), xx, :] = torch.tensor([0.45, 0.45, 0.45])
            img[max(0, yi - 3) : min(height, yi + 4), max(0, xi - 3) : min(width, xi + 4), :] = torch.tensor([0.30, 0.30, 0.30])
            prev_x, prev_y = xi, yi

    return img.clamp(0.0, 1.0).unsqueeze(0)


def _build_report_md(data: Dict) -> str:
    meta = data["meta"]
    warnings = data["warnings"]
    lines: List[str] = []
    lines.append("# LTXV Temporal Envelope Inspector v1.1")
    lines.append("")
    lines.append("This is a non-patching audit node. It only inspects the temporal strength envelope.")
    lines.append("")
    lines.append("## Timeline")
    lines.append(f"- Rendered frames: `{data['total_frames']}`")
    lines.append(f"- LTX latent frames: `{data['latent_frames']}`")
    lines.append(f"- Temporal compression: `{LTX23_TEMPORAL_COMPRESSION}`")
    lines.append(f"- Source: `{data['timing_source']}`")
    lines.append("")
    lines.append("## Envelope settings")
    lines.append(f"- envelope_mode: `{data['envelope_mode']}`")
    lines.append(f"- effect_region: `{data['effect_region']}`")
    lines.append(f"- ramp_mode: `{data['ramp_mode']}`")
    lines.append(f"- ramp_q: `{data['ramp_q']}`")
    lines.append(f"- transition_frames: `{data['transition_frames']}`")
    lines.append(f"- strength_before / during / after: `{data['strength_before']:.6g}` / `{data['strength_during']:.6g}` / `{data['strength_after']:.6g}`")
    if data["envelope_mode"] not in ("transition_frames", "v1.0rc1_transition_frames"):
        lines.append(f"- outside_strength: `{data['outside_strength']:.6g}`")
    lines.append("")
    lines.append("## Resolved regions")
    lines.append(f"- Active region: `{_range_text(meta['region_start'], meta['region_end'])}`")
    if "before_neighbor_start" in meta and "before_neighbor_end" in meta:
        lines.append(f"- Preceding neighbor/ramp segment: `{_range_text(meta['before_neighbor_start'], meta['before_neighbor_end'])}`")
    if "after_neighbor_start" in meta and "after_neighbor_end" in meta:
        lines.append(f"- Following neighbor/ramp segment: `{_range_text(meta['after_neighbor_start'], meta['after_neighbor_end'])}`")
    if data["envelope_mode"] in ("transition_frames", "v1.0rc1_transition_frames"):
        lines.append(f"- v1.0 transition split: `{meta['transition_split_left']}` frames before boundary, `{meta['transition_split_right']}` frames after boundary")
        lines.append(f"- Entry fade: `{meta['entry_fade']}`")
        lines.append(f"- Exit fade: `{meta['exit_fade']}`")
    lines.append("")
    lines.append("## Profile stats")
    lines.append(f"- Frame profile range: `{data['frame_min']:.6g}` .. `{data['frame_max']:.6g}`")
    lines.append(f"- Latent profile range: `{data['latent_min']:.6g}` .. `{data['latent_max']:.6g}`")
    lines.append(f"- Exact full-strength/during frames: `{data['during_plateau_frames']}`")
    lines.append("")
    lines.append("## Q semantics")
    lines.append("- `q=1.0` is linear.")
    lines.append("- `q>1.0` is front-loaded: steeper change at the beginning of the ramp segment.")
    lines.append("- `q<1.0` is delayed: slower beginning, stronger change toward the end.")
    lines.append("- Formula in this inspector: `mix = 1 - (1 - t) ** q`.")
    lines.append("- `flat` ignores Q.")
    lines.append("")
    if warnings:
        lines.append("## Warnings / notes")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Sampled frame-profile values")
    lines.append("")
    lines.append("| frame | position | strength |")
    lines.append("|---:|---:|---:|")
    for row in data["samples"]:
        lines.append(f"| {row['frame']} | {row['pos']:.4f} | {row['value']:.6g} |")
    lines.append("")
    lines.append("## Interpretation")
    if data["envelope_mode"] in ("transition_frames", "v1.0rc1_transition_frames"):
        lines.append("This mode reproduces the published v1.0rc1 boundary-crossfade model.")
    elif data["envelope_mode"] == "v1.1_local":
        lines.append("This mode prototypes the planned local neighbor-segment envelope. Areas beyond the immediate neighboring segments use outside_strength.")
    else:
        lines.append("This mode prototypes the planned hold_strength behavior. Before/after strengths are held beyond the immediate ramp segments.")
    return "\n".join(lines)


class LTXVTemporalEnvelopeInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT", {
                    "tooltip": "Final video-only LTX LATENT used only for timing inspection. Connect before audio/video concat. This node passes it through unchanged."
                }),
                "envelope_mode": (ENVELOPE_MODES, {
                    "default": "v1.1_local",
                    "tooltip": "transition_frames uses before/during/after zones with boundary crossfades. v1.1_local and v1.1_hold_strength prototype the envelope concept."
                }),
                "effect_region": (list(EFFECT_REGIONS.keys()), {
                    "default": "middle third",
                    "tooltip": "Active region controlled by strength_during. The inspector resolves exact frame boundaries from the video_latent timeline."
                }),
                "strength_before": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Target LoRA strength before the active region. In v1.1_local this means the immediate preceding neighbor segment; in hold_strength it is also held earlier."
                }),
                "strength_during": ("FLOAT", {
                    "default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Target LoRA strength inside the selected active region."
                }),
                "strength_after": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Target LoRA strength after the active region. In v1.1_local this means the immediate following neighbor segment; in hold_strength it is also held later."
                }),
                "outside_strength": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Used only in v1.1_local. Strength outside the immediate before/active/after envelope area. Ignored by v1.0rc1 and hold_strength modes."
                }),
                "ramp_mode": (RAMP_MODES, {
                    "default": "flat",
                    "tooltip": "flat uses hard segment strengths and ignores ramp_q. q_curve uses a power/ease-out curve over the full neighboring segment length."
                }),
                "ramp_q": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05,
                    "tooltip": "Used only by q_curve. q=1 linear; q>1 front-loaded/steeper at the beginning; q<1 delayed/slower start."
                }),
                "transition_frames": ("INT", {
                    "default": 16, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Used only by transition_frames mode. Ignored in v1.1 envelope modes, where the segment itself is the ramp duration."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING", "IMAGE", "LTXV_ENVELOPE_DATA")
    RETURN_NAMES = ("video_latent", "report_md", "report_json", "profile_plot", "data")
    FUNCTION = "inspect"
    CATEGORY = "LTXV/Diagnostics"
    DESCRIPTION = "v1.1: inspect LTXV temporal strength envelopes and output reusable envelope data without patching the model."

    def inspect(
        self,
        video_latent,
        envelope_mode: str,
        effect_region: str,
        strength_before: float,
        strength_during: float,
        strength_after: float,
        outside_strength: float,
        ramp_mode: str,
        ramp_q: float,
        transition_frames: int,
    ):
        total_frames, latent_frames, timing_source = _resolve_timing(video_latent)

        if envelope_mode in ("transition_frames", "v1.0rc1_transition_frames"):
            frame_profile, meta = _build_v10_profile(
                total_frames=total_frames,
                effect_region=effect_region,
                strength_before=float(strength_before),
                strength_during=float(strength_during),
                strength_after=float(strength_after),
                transition_frames=int(transition_frames),
            )
        elif envelope_mode == "v1.1_local":
            frame_profile, meta = _build_v11_profile(
                total_frames=total_frames,
                effect_region=effect_region,
                strength_before=float(strength_before),
                strength_during=float(strength_during),
                strength_after=float(strength_after),
                outside_strength=float(outside_strength),
                ramp_mode=ramp_mode,
                ramp_q=float(ramp_q),
                hold_strength=False,
            )
        elif envelope_mode == "v1.1_hold_strength":
            frame_profile, meta = _build_v11_profile(
                total_frames=total_frames,
                effect_region=effect_region,
                strength_before=float(strength_before),
                strength_during=float(strength_during),
                strength_after=float(strength_after),
                outside_strength=float(outside_strength),
                ramp_mode=ramp_mode,
                ramp_q=float(ramp_q),
                hold_strength=True,
            )
        else:
            raise ValueError(f"Unknown envelope_mode: {envelope_mode}")

        latent_profile = _frame_profile_to_latent_profile(frame_profile, LTX23_TEMPORAL_COMPRESSION)
        warnings = _build_warnings(
            envelope_mode=envelope_mode,
            ramp_mode=ramp_mode,
            ramp_q=float(ramp_q),
            transition_frames=int(transition_frames),
            meta=meta,
            total_frames=total_frames,
            strength_before=float(strength_before),
            strength_during=float(strength_during),
            strength_after=float(strength_after),
        )

        data = {
            "version": "LTXV Temporal Envelope Inspector v1.1",
            "total_frames": int(total_frames),
            "latent_frames": int(latent_frames),
            "temporal_compression": int(LTX23_TEMPORAL_COMPRESSION),
            "timing_source": timing_source,
            "envelope_mode": envelope_mode,
            "effect_region": effect_region,
            "strength_before": float(strength_before),
            "strength_during": float(strength_during),
            "strength_after": float(strength_after),
            "outside_strength": None,
            "ramp_mode": ramp_mode,
            "ramp_q": float(ramp_q),
            "transition_frames": int(transition_frames),
            "meta": meta,
            "frame_min": float(frame_profile.min().item()),
            "frame_max": float(frame_profile.max().item()),
            "latent_min": float(latent_profile.min().item()),
            "latent_max": float(latent_profile.max().item()),
            "during_plateau_frames": _plateau_frames(frame_profile, float(strength_during)),
            "samples": _profile_samples(frame_profile),
            "warnings": warnings,
            "frame_profile": [round(float(x), 6) for x in frame_profile.tolist()],
            "latent_profile": [round(float(x), 6) for x in latent_profile.tolist()],
        }

        report_md = _build_report_md(data)
        report_json = json.dumps(data, indent=2, ensure_ascii=False)
        profile_plot = _make_plot_image(frame_profile, latent_profile)
        return (video_latent, report_md, report_json, profile_plot, data)


NODE_CLASS_MAPPINGS = {
    "LTXVTemporalEnvelopeInspector": LTXVTemporalEnvelopeInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVTemporalEnvelopeInspector": "LTXV Temporal Envelope Inspector (v1.1)",
}
