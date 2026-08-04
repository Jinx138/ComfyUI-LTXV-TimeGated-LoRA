# Changelog

## [1.3.0] - 2026-08-04

### Added

- **Temporal Director** workflow with one shared `LTXV Schedule Sync` timeline for up to four semantic prompt segments.
- `LTXV Prompt Relay Encode (Scheduled)` adapter for ComfyUI-PromptRelay.
- `LTXV Time-Gated LoRA (Scheduled)` with strict target-segment selection, separate `q_in` / `q_out`, and independent incoming/outgoing phase shifts.
- `LTXV Temporal Schedule Preview` with seconds-first timing, segment text, warnings, and up to four envelope overlays.
- `LTXV Sigma Tail / Trim` for trimming native ComfyUI `SIGMAS` without flattening the scheduler curve.
- `NOTICE.md` with Prompt Relay attribution and integration details.

### Changed

- PromptRelay epsilon-derived timing can drive Scheduled TimeGate q-curve softness through `relay_timing`.
- `q_in_shift` and `q_out_shift` now travel relative to segment geometry: inward extrema meet at the target midpoint and can form a centered LoRA peak.
- First and final targets use full, equally long virtual neighboring segments, so edge curves match internal-segment geometry and are clipped without phase compression.
- Scheduled envelopes are assembled explicitly as `strength_before → incoming ramp → strength_during → outgoing ramp → strength_after`.
- Temporal Schedule Preview shows duration before frame counts and allows up to 640 prompt characters per segment.
- Schedule Sync recovers missing `|` separators from ordered `segment` / `scene` / `shot` labels or from 2–4 blank-line paragraphs.

### Fixed

- Prevented `strength_after` or `strength_before` baselines from overriding shifted q-curves before the corresponding transition begins.
- Avoided broad `sys.modules` probing in the PromptRelay adapter, preventing imports from triggering unrelated lazy-extension errors.
- Preserved complete q-curve phase when first/last-segment transitions extend outside the visible clip.

### Compatibility

- The established `LTXV Time-Gated LoRA (LTX 2.3)` node and its internal class key remain available.
- Scheduled PromptRelay support requires ComfyUI-PromptRelay; the legacy TimeGate nodes do not.
- Visual LTX 2.3 LoRA gating remains the supported scope; temporal audio LoRA gating is not included.

## [1.1.0]

- Added local/hold-strength envelope modes and q-curve ramps.
- Added reusable envelope data and the CPU-rendered curve preview.
- Improved stacked-node preview handling and release documentation.

## [1.0.0]

- First production release of the stackable LTX 2.3 Time-Gated LoRA node.
- Added effect-region and manual-frame scheduling with video-latent timing.

## [0.7.1]

- Retained diagnostic validation build used to verify localized and stacked temporal patching.
