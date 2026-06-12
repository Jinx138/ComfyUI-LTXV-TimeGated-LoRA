# Release Notes — v1.1

## Highlights

- Added v1.1 envelope modes: `local`, `hold_strength`, and `transition_frames`.
- Added `flat` and `q_curve` ramp behavior.
- Added a `data` output from the Time-Gated LoRA node.
- Added CPU Curve Preview / Analyze node with `data` and optional `data_2` overlay.
- `data_2` can be rendered alone when the first data input is bypassed or missing.
- Preview renders at 1200×800 with larger typography.
- Supports stacked Time-Gated LoRA workflows with visual curve comparison.
- Keeps the v1.0-style transition behavior available as `transition_frames`.

## Recommended release title

`v1.1 — Curve Preview Release`

## Notes

- Existing v1.0-style behavior is preserved via `transition_frames`.
- The Curve Preview visualizes local/relative envelope strength. `lora_strength` is shown as context and does not change the default 0–1 local envelope axis unless local values exceed that range.
- Stochastic detail LoRAs such as rain, snow, film grain, water ripples, and particle effects may be less suitable for long gradual curves than semantic or stylistic transformations.
