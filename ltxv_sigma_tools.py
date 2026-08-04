# SPDX-License-Identifier: MIT
"""Small SIGMAS utilities for LTXV iterative passes.

The tools in this file deliberately operate on ComfyUI's native SIGMAS tensor.
They avoid SIGMAS->float conversion so they remain useful when external helper
nodes fail to stringify/convert scheduler outputs.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

try:
    import torch
except Exception:  # pragma: no cover - Comfy normally has torch
    torch = None


_VERSION = "v1.3.0"


def _to_1d_sigmas(sigmas: Any):
    if torch is not None and isinstance(sigmas, torch.Tensor):
        return sigmas.flatten()
    # Some custom nodes may wrap tensors in lists/tuples.
    if isinstance(sigmas, (list, tuple)) and sigmas:
        if torch is not None:
            return torch.tensor([float(x) for x in sigmas], dtype=torch.float32)
        return [float(x) for x in sigmas]
    raise ValueError("Expected SIGMAS tensor or a non-empty numeric sequence.")


def _numel(x: Any) -> int:
    if torch is not None and isinstance(x, torch.Tensor):
        return int(x.numel())
    return len(x)


def _value(x: Any, idx: int) -> float:
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x[idx].detach().cpu().item())
    return float(x[idx])


def _slice(x: Any, start: int, end: int | None = None):
    return x[start:end]


def _cat_zero_if_needed(x: Any, zero_like_source: Any):
    n = _numel(x)
    if n > 0 and abs(_value(x, n - 1)) < 1e-8:
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        zero = torch.zeros((1,), dtype=x.dtype, device=x.device)
        return torch.cat([x, zero], dim=0)
    return list(x) + [0.0]


def _csv(x: Any, precision: int = 4) -> str:
    vals = [_value(x, i) for i in range(_numel(x))]
    return ", ".join(f"{v:.{precision}f}" for v in vals)


class LTXVSigmaTail:
    """Trim scheduler SIGMAS without converting them to floats first."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "mode": ([
                    "drop_first_count",
                    "keep_last_count",
                    "keep_last_fraction",
                    "start_at_or_below_sigma",
                ], {"default": "drop_first_count"}),
                "drop_first_count": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "keep_last_count": ("INT", {"default": 9, "min": 2, "max": 128, "step": 1}),
                "keep_last_fraction": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 1.00, "step": 0.01}),
                "max_start_sigma": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.0001}),
                "ensure_zero_end": ("BOOLEAN", {"default": True}),
                "csv_precision": ("INT", {"default": 4, "min": 2, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "STRING", "STRING")
    RETURN_NAMES = ("sigmas", "sigmas_csv", "info")
    FUNCTION = "trim"
    CATEGORY = "LTXV/Timing"
    DESCRIPTION = "Trim a SIGMAS tensor for 2nd/3rd pass experiments. Useful after BasicScheduler: keep the tail of a linear_quadratic schedule without using SIGMAS-to-float nodes."

    def trim(
        self,
        sigmas,
        mode: str,
        drop_first_count: int,
        keep_last_count: int,
        keep_last_fraction: float,
        max_start_sigma: float,
        ensure_zero_end: bool,
        csv_precision: int,
    ):
        s = _to_1d_sigmas(sigmas)
        n = _numel(s)
        if n < 2:
            raise ValueError("SIGMAS must contain at least two values.")

        has_terminal_zero = abs(_value(s, n - 1)) < 1e-8
        nonterminal_n = n - 1 if has_terminal_zero else n
        start = 0
        reason = ""

        if mode == "drop_first_count":
            start = int(max(0, min(drop_first_count, max(0, n - 2))))
            reason = f"dropped first {start} sigma value(s)"
        elif mode == "keep_last_count":
            keep = int(max(2, min(keep_last_count, n)))
            start = max(0, n - keep)
            reason = f"kept last {keep} value(s), terminal zero included if present"
        elif mode == "keep_last_fraction":
            frac = float(max(0.05, min(1.0, keep_last_fraction)))
            keep_nonterminal = int(max(1, round(nonterminal_n * frac)))
            start = max(0, nonterminal_n - keep_nonterminal)
            reason = f"kept last {frac:.2f} of non-terminal sigma values ({keep_nonterminal}/{nonterminal_n})"
        elif mode == "start_at_or_below_sigma":
            threshold = float(max_start_sigma)
            start = 0
            for i in range(nonterminal_n):
                if _value(s, i) <= threshold:
                    start = i
                    break
            else:
                start = max(0, nonterminal_n - 1)
            reason = f"started at first sigma <= {threshold:.4f}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out = _slice(s, start, None)
        if ensure_zero_end:
            out = _cat_zero_if_needed(out, s)

        out_n = _numel(out)
        if out_n < 2:
            raise ValueError("Trimmed SIGMAS would contain fewer than two values; adjust trim settings.")

        csv = _csv(out, int(csv_precision))
        info = (
            f"LTXV Sigma Tail {_VERSION}\n"
            f"mode={mode}\n"
            f"input_count={n} output_count={out_n}\n"
            f"start_index={start}\n"
            f"input_first={_value(s,0):.6f} input_last={_value(s,n-1):.6f}\n"
            f"output_first={_value(out,0):.6f} output_last={_value(out,out_n-1):.6f}\n"
            f"{reason}\n"
            f"sigmas={csv}"
        )
        return (out, csv, info)


NODE_CLASS_MAPPINGS = {
    "LTXVSigmaTail": LTXVSigmaTail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVSigmaTail": "LTXV Sigma Tail / Trim",
}
