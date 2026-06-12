# Changelog

## v1.0rc1

- Productive release candidate based on the validated stackable v0.7 runtime path.
- Removed the visible `diagnostic_probe` operation from the production node; v0.7.1 remains the archived validation build.
- Added unchanged `video_latent` passthrough output for clean multi-node stacking.
- Changed visible strength slider increments to `0.05` for practical mouse adjustment.
- Kept mode-dependent greying of inactive schedule inputs and refined tooltips.
- Condensed `schedule_info` for production use.
- Expanded README with placement, PromptRelay equal-segment workflow, triggerword guidance, I2V-anchor troubleshooting, LTX strength-range guidance, memory/compile limits and validated demo scenarios.

## v0.7.1 — diagnostic validation build

- Added a deterministic in-model temporal diagnostic probe.
- Successfully validated temporally localized single-region probing and two stacked probes in separated quarter regions.
- Retained separately for future regression testing; not part of the v1.0rc1 production UI.

## v0.7

- Added safe chaining of multiple `LTXV Time-Gated LoRA (LTX 2.3)` nodes.
- Fixed the false persistent-wrapper contamination error when stacking temporal nodes.

## v0.6

- Renamed the visible node to `LTXV Time-Gated LoRA (LTX 2.3)`.
- Made `video_latent` a required timing input and removed redundant `total_frames`.

## v0.5

- Simplified outputs to `model` and `schedule_info`.
- Removed visible audio and temporal-compression controls; audio time-gating left outside v1.0 scope.

## v0.4

- Added `effect_region`, optional latent timing and the low-rank VRAM-optimized gating path.


## v1.1rc5-dev
- Preview/Analyze layout polish: narrower canvas, larger fonts, dedicated legend row, improved truncation.


## v1.1rc6-dev
- Preview/Analyze layout polish: taller 900x600 canvas and reflowed header to avoid overlap.

## v1.1rc7-dev
- Local envelope before/after levels now hold outside the immediate ramp window.
- Removed obsolete local baseline hard-step warnings.
- Preview typography adjusted closer to ComfyUI system sizing.

## v1.1rc8-dev
- Curve Preview Y-axis no longer expands below 0 when values only exceed 1.0 upward.
- Curve Preview fonts increased slightly for readability.

## v1.1
- Preview now shows thicker frame-envelope curves only; latent-average overlay is hidden for release clarity.
- Renamed transition-frame envelope mode to `transition_frames` while keeping the old value accepted internally for compatibility.

## v1.1
- Curve Preview subtitle now shows separate, equal summary lines for data and data_2.

## v1.1
- Curve Preview overlay header now shows mode/region/ramp per data input instead of a data-only global mode line.

## v1.1
- Curve Preview / Analyze now accepts data_2 alone when data is missing/bypassed.
- data/data_2 are optional with a clear error only if both are absent.

## v1.1
- Curve Preview / Analyze default image size increased to 1200x800.
- No behavior change to the data/data_2 fallback logic from v1.1.

## v1.1
- Curve Preview / Analyze typography increased for 1200x800 output.
- Header/legend spacing adjusted and test renders generated.

## v1.1
- Removed the `ramp_q ignored in flat mode` warning from generated warnings and preview display.
