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
