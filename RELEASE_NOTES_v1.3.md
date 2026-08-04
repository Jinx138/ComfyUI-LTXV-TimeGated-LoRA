# ComfyUI-LTXV-TimeGated-LoRA v1.3.0

v1.3 adds a shared-schedule **Temporal Director** beside the established Time-Gated LoRA node.

## Highlights

- One `LTXV Schedule Sync` timeline drives prompt segmentation, Scheduled TimeGates, and preview output.
- Missing `|` separators can be reconstructed from ordered segment/scene/shot labels or blank-line paragraphs.
- PromptRelay epsilon can provide the shared q-curve softness timing.
- Independent `q_in_shift` and `q_out_shift` controls support early/late transitions and centered LoRA peaks.
- First and final segments use symmetric virtual neighbors, so off-screen ramps retain the same geometry as internal transitions.
- The schedule preview is seconds-first, supports up to four envelopes, and can display up to 640 characters per segment.
- `LTXV Sigma Tail / Trim` keeps native scheduler sigmas for controlled later-pass denoising.

## New nodes

- `LTXV Schedule Sync`
- `LTXV Prompt Relay Encode (Scheduled)`
- `LTXV Time-Gated LoRA (Scheduled)`
- `LTXV Temporal Schedule Preview`
- `LTXV Sigma Tail / Trim`

## Upgrade notes

- Restart ComfyUI and hard-refresh the browser after installation.
- Add fresh Scheduled node instances when upgrading from development builds because the widget layout changed during development.
- Existing workflows using `LTXV Time-Gated LoRA (LTX 2.3)` remain supported.
- The Scheduled PromptRelay adapter requires an installed and enabled `ComfyUI-PromptRelay`.

## Scope

- LTX 2.3 visual LoRA gating.
- Temporal audio LoRA gating is not included.
- `torch.compile` remains unsupported for the runtime-gated path.
