"""
LTXV Envelope Curve Preview / Analyze v1.1

CPU-side visual preview node for LTXV temporal envelope data.

The preview intentionally plots the local/relative envelope, not the absolute
patch strength multiplied by lora_strength.  The lora_strength value is shown
as context in the title; the Y axis defaults to 0..1 and only expands when the
local envelope itself goes below 0 or above 1.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - fallback exists for environments without PIL
    Image = None
    ImageDraw = None
    ImageFont = None


# Comfy-ish dark palette. Kept deliberately neutral/readable.
BG = (30, 33, 38)
PANEL = (36, 39, 45)
PLOT_BG = (40, 43, 50)
GRID = (73, 77, 87)
AXIS = (150, 156, 166)
TEXT = (222, 226, 232)
MUTED = (165, 171, 182)
WARNING = (238, 154, 99)
PRIMARY = (238, 238, 238)
PRIMARY_LATENT = (230, 164, 75)
SECONDARY = (123, 190, 255)
SECONDARY_LATENT = (205, 136, 255)
BAND_BEFORE = (44, 54, 70)
BAND_ACTIVE = (46, 64, 50)
BAND_AFTER = (70, 54, 44)
BAND_OUTSIDE = (35, 37, 42)


def _as_data_dict(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return json.loads(data)
    raise ValueError(f"Expected LTXV_ENVELOPE_DATA dict or JSON string, got {type(data).__name__}")


def _maybe_data_dict(data: Any) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    if isinstance(data, dict) and not data:
        return None
    return _as_data_dict(data)


def _slot_label(data: Dict[str, Any], default: str) -> str:
    return str(data.get("_preview_slot", default) or default)


def _to_float_list(values: Any, name: str, required: bool = True) -> List[float]:
    if values is None:
        if required:
            raise ValueError(f"Envelope data is missing '{name}'")
        return []
    if isinstance(values, torch.Tensor):
        return [float(x) for x in values.detach().cpu().flatten().tolist()]
    return [float(x) for x in values]


def _local_axis_range(profiles: Iterable[List[float]]) -> Tuple[float, float]:
    vals: List[float] = []
    for profile in profiles:
        vals.extend(float(v) for v in profile)
    if not vals:
        return 0.0, 1.0
    vmin = min(vals)
    vmax = max(vals)
    # Local-envelope convention: default reference range is 0..1.
    if vmin >= 0.0 and vmax <= 1.0:
        return 0.0, 1.0

    # If the envelope only exceeds 1.0 upward, keep the lower bound anchored at 0.
    # If it only exceeds downward, keep the upper bound anchored at 1.
    if vmin >= 0.0 and vmax > 1.0:
        hi = vmax
        pad = max(0.04, (hi - 0.0) * 0.06)
        return 0.0, hi + pad
    if vmin < 0.0 and vmax <= 1.0:
        lo = vmin
        pad = max(0.04, (1.0 - lo) * 0.06)
        return lo - pad, 1.0

    # If it exceeds on both sides, expand both bounds.
    lo = min(0.0, vmin)
    hi = max(1.0, vmax)
    if abs(hi - lo) < 1e-8:
        return lo - 0.1, hi + 0.1
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def _pil_to_comfy(img: "Image.Image") -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _get_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_text(draw, xy: Tuple[int, int], text: str, fill=TEXT, font=None, anchor: Optional[str] = None):
    try:
        draw.text(xy, text, fill=fill, font=font, anchor=anchor)
    except TypeError:
        draw.text(xy, text, fill=fill, font=font)


def _line_points_for_profile(values: List[float], x_of_frame, y_of_value) -> List[Tuple[int, int]]:
    n = len(values)
    if n <= 0:
        return []
    if n == 1:
        return [(x_of_frame(0), y_of_value(values[0]))]
    return [(x_of_frame(i), y_of_value(values[i])) for i in range(n)]


def _draw_profile_line(draw, points: List[Tuple[int, int]], fill, width: int = 3):
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=fill)
        return
    if len(points) > 1:
        try:
            draw.line(points, fill=fill, width=width, joint="curve")
        except TypeError:
            draw.line(points, fill=fill, width=width)


def _frame_count(data: Dict[str, Any], profile: List[float]) -> int:
    n = int(data.get("total_frames", len(profile) or 1))
    return max(1, n)


def _fps_from_data(data: Dict[str, Any]) -> float:
    fps = data.get("fps", data.get("preview_fps", 24.0))
    try:
        fps = float(fps)
    except Exception:
        fps = 24.0
    return fps if fps > 0 else 24.0


def _lora_strength_label(data: Dict[str, Any]) -> str:
    val = data.get("lora_strength", data.get("strength_model", None))
    if val is None:
        return "lora_strength (n/a)"
    try:
        return f"lora_strength ({float(val):.2f})"
    except Exception:
        return f"lora_strength ({val})"


def _mode_label(data: Dict[str, Any]) -> str:
    ramp_mode = data.get("ramp_mode", "")
    try:
        q = float(data.get("ramp_q", 1.0))
        q_part = f" q={q:.2f}" if ramp_mode == "q_curve" else ""
    except Exception:
        q_part = ""
    return f"{data.get('envelope_mode')} / {data.get('effect_region')} / {ramp_mode}{q_part}"


def _display_name(data: Dict[str, Any], default: str) -> str:
    name = str(data.get("lora_name", "") or "").split("/")[-1]
    return name if name else default


def _truncate_label(text: str, max_chars: int = 22) -> str:
    text = str(text or '')
    return text if len(text) <= max_chars else text[: max(1, max_chars - 1)] + '…'




def _profile_range(profile: List[float]) -> Tuple[float, float]:
    if not profile:
        return 0.0, 0.0
    return float(min(profile)), float(max(profile))

def _combined_warnings(data: Dict[str, Any], data_2: Optional[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    warnings.extend(str(w) for w in (data.get("warnings", []) or []))
    if data_2 is not None:
        warnings.extend(f"data_2: {w}" for w in (data_2.get("warnings", []) or []))
        n1 = int(data.get("total_frames", 0) or 0)
        n2 = int(data_2.get("total_frames", 0) or 0)
        if n1 and n2 and n1 != n2:
            warnings.append(f"Timeline mismatch: data has {n1} frames, data_2 has {n2} frames. Overlay may be misleading.")
    # De-duplicate while preserving order.
    out = []
    seen = set()
    for w in warnings:
        if "ramp_q ignored in flat mode" in str(w):
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _render_with_pil(
    data: Dict[str, Any],
    data_2: Optional[Dict[str, Any]],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    frame_profile = _to_float_list(data.get("frame_profile"), "frame_profile")
    latent_profile = _to_float_list(data.get("latent_profile", []), "latent_profile", required=False)
    frame_profile_2: List[float] = []
    latent_profile_2: List[float] = []
    if data_2 is not None:
        frame_profile_2 = _to_float_list(data_2.get("frame_profile"), "data_2.frame_profile")
        latent_profile_2 = _to_float_list(data_2.get("latent_profile", []), "data_2.latent_profile", required=False)

    total_frames = _frame_count(data, frame_profile)
    fps = _fps_from_data(data)
    meta = data.get("meta", {}) or {}
    warnings = _combined_warnings(data, data_2)

    width = int(max(720, min(2048, width)))
    height = int(max(320, min(1024, height)))

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    font_small = _get_font(19)
    font = _get_font(22)
    font_bold = _get_font(23, bold=True)
    font_title = _get_font(28, bold=True)

    margin = 16
    left = 72
    right = 24
    top = 186
    bottom = 118 if warnings else 86
    plot_w = max(1, width - left - right - margin)
    plot_h = max(1, height - top - bottom)
    x0, y0 = left, top
    x1, y1 = left + plot_w, top + plot_h

    # Panel and plot background.
    draw.rectangle((0, 0, width, height), fill=BG)
    draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=10, fill=PANEL, outline=(54, 58, 66), width=1)
    draw.rectangle((x0, y0, x1, y1), fill=PLOT_BG)

    values_for_scale = list(frame_profile) + list(frame_profile_2)
    vmin, vmax = _local_axis_range([values_for_scale])

    def x_of_frame(frame_idx: int, total: Optional[int] = None) -> int:
        n = max(1, int(total if total is not None else total_frames))
        if n <= 1:
            return x0
        return int(round(x0 + (max(0, min(n - 1, frame_idx)) / float(n - 1)) * plot_w))

    def x_of_boundary(frame_idx: int, total: Optional[int] = None) -> int:
        n = max(1, int(total if total is not None else total_frames))
        return int(round(x0 + (max(0, min(n, frame_idx)) / float(n)) * plot_w))

    def y_of_value(v: float) -> int:
        norm = (float(v) - vmin) / max(1e-8, (vmax - vmin))
        return int(round(y1 - norm * plot_h))

    # Segment bands from primary data.
    region_start = int(meta.get("region_start", 0))
    region_end = int(meta.get("region_end", 0))
    before_start = int(meta.get("before_neighbor_start", region_start))
    before_end = int(meta.get("before_neighbor_end", region_start))
    after_start = int(meta.get("after_neighbor_start", region_end))
    after_end = int(meta.get("after_neighbor_end", region_end))

    def band(start: int, end: int, fill):
        if end <= start:
            return
        xa = x_of_boundary(start)
        xb = x_of_boundary(end)
        draw.rectangle((xa, y0, xb, y1), fill=fill)

    # Outside first, then local/active regions.
    band(0, total_frames, BAND_OUTSIDE)
    band(before_start, before_end, BAND_BEFORE)
    band(region_start, region_end, BAND_ACTIVE)
    band(after_start, after_end, BAND_AFTER)

    # Grid and axes.
    for i in range(0, 9):
        x = x0 + int(round(i * plot_w / 8.0))
        fill = AXIS if i in (0, 8) else GRID
        draw.line((x, y0, x, y1), fill=fill, width=1)
    # y ticks include 0..1 when possible; expanded ranges still get 5 ticks.
    for i in range(0, 5):
        frac = i / 4.0
        v = vmin + (vmax - vmin) * frac
        y = y_of_value(v)
        fill = AXIS if i in (0, 4) else GRID
        draw.line((x0, y, x1, y), fill=fill, width=1)
        _draw_text(draw, (14, y - 8), f"{v:.2g}", fill=MUTED, font=font_small)
    draw.rectangle((x0, y0, x1, y1), outline=AXIS, width=1)

    # 0 and 1 reference lines if they are visible and not already on edge.
    for ref_v, label in [(0.0, "0"), (1.0, "1")]:
        if vmin <= ref_v <= vmax:
            y = y_of_value(ref_v)
            draw.line((x0, y, x1, y), fill=(105, 111, 122), width=1)
            if label == "1" and abs(y - y0) > 3:
                _draw_text(draw, (x1 - 18, y - 14), "1.0", fill=MUTED, font=font_small)

    # Boundary lines and labels.
    boundary_items = [
        (before_start, "before"),
        (region_start, "active"),
        (region_end, "after"),
        (after_end, "end"),
    ]
    seen = set()
    for frame_idx, label in boundary_items:
        frame_idx = max(0, min(total_frames, int(frame_idx)))
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        x = x_of_boundary(frame_idx)
        draw.line((x, y0, x, y1), fill=(126, 132, 144), width=1)
        _draw_text(draw, (x + 4, y1 + 6), f"{label}\n{frame_idx}", fill=MUTED, font=font_small)

    # Main curves.
    frame_points = _line_points_for_profile(frame_profile, lambda i: x_of_frame(i, total_frames), y_of_value)
    _draw_profile_line(draw, frame_points, fill=PRIMARY, width=6)

    if data_2 is not None and frame_profile_2:
        total_frames_2 = _frame_count(data_2, frame_profile_2)
        pts2 = _line_points_for_profile(frame_profile_2, lambda i: x_of_frame(i, total_frames_2), y_of_value)
        _draw_profile_line(draw, pts2, fill=SECONDARY, width=6)
    # Titles and summary.
    seconds = (total_frames - 1) / fps if fps > 0 else 0.0
    title = "LTXV Envelope Curve"
    _draw_text(draw, (left, 24), title, fill=TEXT, font=font_title)
    summary = f"{total_frames} frames / {data.get('latent_frames')} latent / {seconds:.2f}s @ {fps:.2f} fps"
    _draw_text(draw, (x1, 30), summary, fill=MUTED, font=font, anchor="ra")
    data_min, data_max = _profile_range(frame_profile)

    data_label = _slot_label(data, "data")
    if data_2 is not None and frame_profile_2:
        # In overlay mode, data and data_2 are peers: each line carries its own
        # mode/region/ramp info, local envelope range and lora_strength.
        data2_label = _slot_label(data_2, "data_2")
        data2_min, data2_max = _profile_range(frame_profile_2)
        subtitle_1 = f"{data_label}: {_mode_label(data)} · env {data_min:.2g}..{data_max:.2g} · {_lora_strength_label(data)}"
        subtitle_2 = f"{data2_label}: {_mode_label(data_2)} · env {data2_min:.2g}..{data2_max:.2g} · {_lora_strength_label(data_2)}"
        _draw_text(draw, (left, 70), subtitle_1, fill=MUTED, font=font)
        _draw_text(draw, (left, 100), subtitle_2, fill=MUTED, font=font)
        legend_y = 136
    else:
        prefix = f"{data_label}: " if data_label != "data" else ""
        mode_line = f"{prefix}{_mode_label(data)} · env {data_min:.2g}..{data_max:.2g} · {_lora_strength_label(data)}"
        _draw_text(draw, (left, 70), mode_line, fill=TEXT, font=font_bold)
        legend_y = 106

    # Legend in dedicated header row to avoid overlap with summary.
    legend_x = left

    def legend_item(x, y, color, label, text_fill=TEXT):
        draw.rounded_rectangle((x, y + 5, x + 22, y + 17), radius=3, fill=color)
        _draw_text(draw, (x + 30, y), _truncate_label(label, 28), fill=text_fill, font=font_small)
        bbox = draw.textbbox((x + 30, y), _truncate_label(label, 28), font=font_small)
        return bbox[2] + 30

    nx = legend_item(legend_x, legend_y, PRIMARY, _display_name(data, "data"))
    if data_2 is not None:
        nx = legend_item(nx + 26, legend_y, SECONDARY, _display_name(data_2, "data_2"))

    # Footer warnings, wrapped.
    if warnings:
        footer_x = left
        footer_y = height - bottom + 28
        warn_text = "Warnings: " + " | ".join(warnings[:4])
        max_chars = max(60, int((width - left - right) / 7.2))
        lines = textwrap.wrap(warn_text, width=max_chars)[:2]
        for i, line in enumerate(lines):
            _draw_text(draw, (footer_x, footer_y + i * 15), line, fill=WARNING, font=font_small)

    return _pil_to_comfy(img)


def _fallback_torch_plot(data: Dict[str, Any], width: int, height: int) -> torch.Tensor:
    values = torch.tensor(_to_float_list(data.get("frame_profile"), "frame_profile"), dtype=torch.float32)
    width = int(max(720, min(2048, width)))
    height = int(max(320, min(1024, height)))
    img = torch.zeros((height, width, 3), dtype=torch.float32) + 0.14
    if values.numel() == 0:
        return img.unsqueeze(0)
    vmin, vmax = _local_axis_range([values.tolist()])
    for xi in range(width):
        idx = int(round(xi * (values.numel() - 1) / max(1, width - 1)))
        norm = (float(values[idx]) - vmin) / max(1e-8, vmax - vmin)
        y = int(round((height - 20) - norm * (height - 40)))
        img[max(0, y - 1): min(height, y + 2), xi, :] = 0.92
    return img.unsqueeze(0)


def _build_summary(data: Dict[str, Any], data_2: Optional[Dict[str, Any]]) -> str:
    warnings = _combined_warnings(data, data_2)
    fps = _fps_from_data(data)
    local_min = float(data.get("frame_min", 0.0))
    local_max = float(data.get("frame_max", 0.0))
    lines = [
        "# LTXV Envelope Curve Preview / Analyze",
        "",
        f"- Source node: `{data.get('source_node', 'unknown')}`",
        f"- Source version: `{data.get('version', 'unknown')}`",
        f"- Mode: `{data.get('envelope_mode')}` / `{data.get('ramp_mode')}` / q=`{float(data.get('ramp_q', 1.0)):.3g}`",
        f"- Region: `{data.get('effect_region')}`",
        f"- Frames: `{data.get('total_frames')}` rendered / `{data.get('latent_frames')}` latent @ `{fps:.3g}` fps",
        f"- Y axis: local envelope strength, default 0..1; expands only for local values outside this range",
        f"- Local envelope range: `{local_min:.6g}` .. `{local_max:.6g}`",
        f"- {_lora_strength_label(data)}",
    ]
    if data_2 is not None:
        lines.extend([
            "",
            "## Overlay data_2",
            f"- Source version: `{data_2.get('version', 'unknown')}`",
            f"- Mode: `{data_2.get('envelope_mode')}` / `{data_2.get('ramp_mode')}` / q=`{float(data_2.get('ramp_q', 1.0)):.3g}`",
            f"- Region: `{data_2.get('effect_region')}`",
            f"- {_lora_strength_label(data_2)}",
        ])
    if warnings:
        lines.append("")
        lines.append("## Warnings / notes")
        for w in warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


class LTXVEnvelopeCurvePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "data": ("LTXV_ENVELOPE_DATA", {
                    "tooltip": "Optional first envelope data stream from LTXV Time-Gated LoRA. If omitted, data_2 can be shown alone."
                }),
                "data_2": ("LTXV_ENVELOPE_DATA", {
                    "tooltip": "Optional second envelope data stream to overlay in the same coordinate system. If data is missing, this input is rendered by itself."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("curve_image", "report_md", "report_json")
    FUNCTION = "render"
    CATEGORY = "LTXV/Diagnostics"
    DESCRIPTION = "v1.1: 1200x800 Curve Preview with larger typography; suppresses low-value flat-mode ramp_q warning; data/data_2 optional fallback retained."

    def render(self, data=None, data_2=None):
        data = _maybe_data_dict(data)
        data_2 = _maybe_data_dict(data_2)

        if data is None and data_2 is None:
            raise ValueError("Connect at least one envelope data input: data or data_2.")

        if data is None:
            # If the upstream first node is bypassed, users may still have the later
            # gate connected to data_2. Promote it for rendering while preserving
            # its visible slot label in the header.
            data = dict(data_2)
            data["_preview_slot"] = "data_2"
            data_2 = None
        else:
            data = dict(data)
            data.setdefault("_preview_slot", "data")
            if data_2 is not None:
                data_2 = dict(data_2)
                data_2.setdefault("_preview_slot", "data_2")

        width = 1200
        height = 800
        if Image is None:
            image = _fallback_torch_plot(data, width, height)
        else:
            image = _render_with_pil(data, data_2, width=width, height=height)
        report_obj = dict(data)
        if data_2 is not None:
            report_obj["data_2"] = data_2
            report_obj["overlay_warnings"] = _combined_warnings(data, data_2)
        report_json = json.dumps(report_obj, indent=2, ensure_ascii=False)
        return (image, _build_summary(data, data_2), report_json)


NODE_CLASS_MAPPINGS = {
    "LTXVEnvelopeCurvePreview": LTXVEnvelopeCurvePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVEnvelopeCurvePreview": "LTXV Envelope Curve Preview / Analyze (CPU)"
}
