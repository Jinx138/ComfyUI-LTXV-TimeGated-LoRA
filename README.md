# ComfyUI-LTXV-TimeGated-LoRA v1.0rc1

Node: **LTXV Time-Gated LoRA (LTX 2.3)**

Temporally apply visual LTX 2.3 LoRAs to selected regions of a single continuous video sampling run. Use it for effects such as a style appearing only in the middle third, a transformation beginning in the final quarter, or multiple independent scheduled LoRAs chained in one clip.

## Scope

- Visual LTX 2.3 LoRA layers only; temporal audio gating is intentionally not supported in v1.0.
- `effect_region` mode for fast placement by halves, thirds or quarters.
- `manual_frames` mode for exact signed multi-segment schedules.
- Timing is derived from the required final **video-only** `video_latent`.
- Multiple instances can be chained additively.
- Low-rank temporal gating includes a safe mode and an experimental lower-VRAM in-place mode.

The validated diagnostic build is kept separately as **v0.7.1**; its diagnostic probe is intentionally not exposed in the production release candidate.

## Placement

Place this node, or a chain of these nodes, **last in the MODEL chain directly before the sampler or guider**.

For a PromptRelay workflow:

```text
... -> PromptRelayEncode -> LTX2SamplingPreviewOverride
    -> LTXV Time-Gated LoRA -> optional additional LTXV Time-Gated LoRA -> SamplerCustom
```

Do not put this runtime-gated node in the normal static LoRA group near the model loader. Static LoRAs and distilled LoRAs may remain in their original upstream locations.

## `video_latent` timing and stacking

Connect the final **video-only** latent after image/FLF conditioning and before any audio/video concat node. The node now passes this latent through unchanged so stacked nodes can be wired cleanly.

```text
Timing path:
LTXVImgToVideoInplaceKJ video_latent
  -> LTXV Time-Gated LoRA A video_latent
  -> LTXV Time-Gated LoRA B video_latent

MODEL path:
LTX2SamplingPreviewOverride MODEL
  -> LTXV Time-Gated LoRA A MODEL
  -> LTXV Time-Gated LoRA B MODEL
  -> SamplerCustom MODEL
```

Do **not** connect an audio+video combined latent to `video_latent`. The node derives the real LTX timeline from the video temporal dimension and reports the resolved timeline in `schedule_info`. LTX 2.3 temporal compression is fixed internally to `8`.

## Inputs and outputs

### Inputs

- `model`: completed LTX 2.3 MODEL path to patch.
- `video_latent`: required final video-only timing reference.
- `lora_name`: compatible visual LTX 2.3 LoRA.
- `strength_model`: global LoRA multiplier. Effective strength is `strength_model × schedule value`.
- `schedule_mode`: `effect_region` or `manual_frames`.
- `effect_region`, `strength_before`, `strength_during`, `strength_after`: used in `effect_region` mode.
- `segment_lengths`, `segment_strengths`: used in `manual_frames` mode.
- `transition_frames`: crossfade width at effect boundaries.
- `memory_mode`: `optimized_safe` by default; `optimized_low_vram_inplace` is experimental.

The numeric strength widgets use practical `0.05` steps for mouse adjustment. `segment_strengths` remains a freely editable string for advanced schedules.

### Outputs

- `model`: patched MODEL path.
- `video_latent`: unchanged timing latent passthrough for stacking.
- `schedule_info`: compact resolved-schedule and compatibility report.

## `effect_region` mode

Example: apply a style only in the middle third.

```text
schedule_mode:     effect_region
effect_region:     middle third
strength_model:    1.50
strength_before:   0
strength_during:   1
strength_after:    0
transition_frames: 16
```

Available regions include quarters, thirds, `first half`, `center half` and `last half`.

## `manual_frames` mode

Use exact rendered-frame segment lengths and multipliers. Signed and decimal values are valid.

```text
schedule_mode:      manual_frames
segment_lengths:    120,193,192
segment_strengths:  0,-1,1.5
```

The segment lengths must sum to the rendered frame length resolved from `video_latent`.

## PromptRelay compatibility

For simple synchronization, leave PromptRelay `segment_lengths` empty. PromptRelay will evenly distribute its `|`-separated local prompts, which matches the region philosophy of this node when the prompt segmentation matches the selected granularity:

- halves: use **2** local prompt segments
- thirds: use **3** local prompt segments
- quarters: use **4** local prompt segments

Examples:

```text
middle third   -> prompt A | prompt B | prompt C
second quarter -> prompt A | prompt B | prompt C | prompt D
last half      -> prompt A | prompt B
```

For stacked nodes using mixed region types, exact PromptRelay synchronization may require manual segment planning.

### Trigger-dependent LoRAs

When a Style LoRA requires a trigger word or activation phrase, include it inside every **active PromptRelay local segment** in which that Time-Gated LoRA is enabled. Do not rely only on the global prompt for a scheduled style reveal. Some broad style LoRAs, such as Claymation-style LoRAs, may respond to a global style description, but may still require higher strength.

## Best practices

### Strength ranges

LTX 2.3 LoRA strengths are not inherently limited to `0–1`. Depending on the LoRA, useful values may exceed `1.0`. Slider or transformation LoRAs may use meaningful negative and positive values, and their two directions do not have to be equally strong.

If an effect is weak, do not assume the LoRA is broken at `1.0`; test appropriate higher strengths while watching for artifacts.

### I2V image anchor can suppress transformations

In I2V workflows, a strong image anchor can substantially suppress visible LoRA transformations. With nodes such as `LTXVImgToVideoInplaceKJ`, a high value such as `strength_1 = 1.0` may pull the appearance back toward the initial image late in denoising, even when early sampler previews already show the LoRA effect.

For strong style or transformation effects, try a lower anchor first, for example `0.2–0.5`, before pushing LoRA strength further.

### Transition frames

- `8–16` frames: good starting point for ordinary style/effect reveals.
- `0` frames: intentionally hard switches or diagnosis.
- `16–32` frames: stronger transformations that need a gentler blend.

At 24 fps, `8–16` frames correspond to approximately `0.33–0.67` seconds.

### Frame rate and valid LTX length

For convenient whole-second durations, prefer `24 fps`: whole seconds map cleanly to the LTX `8n+1` frame convention when the workflow adds the final `+1` frame.

## Memory and compatibility

- Start with `memory_mode = optimized_safe`.
- Try `optimized_low_vram_inplace` only after an OOM in safe mode, especially for long clips or stacked nodes, and compare output stability.
- Each additional scheduled LoRA adds compute and VRAM pressure.
- SageAttention is usable, but keep `allow_compile = false` when using this node.
- `torch.compile` / `TorchCompileModelAdvanced` is not currently supported.
- Temporal audio LoRA gating is not supported in v1.0.
- A real LoRA may influence later video semantics after its active region even when the gate itself is temporally correct.

## Validated demo scenarios

### Age Transformation Reveal

A single continuous I2V run showing an adult appearance, a younger reveal and an elderly reveal using a time-gated age/slider LoRA with three PromptRelay segments. A reduced I2V image anchor was critical for visible transformation.

### Claymation Style Reveal

A realistic scene reveals a Claymation style only in the middle third, then returns to realism. The sign's blank reverse side acts as a natural occluder/reveal transition. This workflow is the recommended release-candidate smoke test.

## Installation

Extract the folder into ComfyUI `custom_nodes`, restart ComfyUI, and add a fresh instance of **LTXV Time-Gated LoRA (LTX 2.3)** to the workflow. Because v1.0rc1 adds a `video_latent` output, replace older node instances instead of relying on stored widget/output indexes.


## Included example workflow: Claymation Style Reveal

File:

```text
workflows/PromptRelay_LTXV_TimeGatedLoRA_Claymation_v1.0rc1.json
```

This release-candidate example is a PromptRelay LTX 2.3 I2V workflow validated with one active **LTXV Time-Gated LoRA (LTX 2.3)** node:

```text
effect_region:     middle third
strength_model:    5.0
strength_before:   0
strength_during:   1
strength_after:    0
transition_frames: 16
memory_mode:       optimized_safe
```

Key settings captured in the validated workflow export:

```text
fps:                       24
sampler:                   euler_ancestral
scheduler:                 linear_quadratic
steps:                     10
cfg:                       1.3
I2V image anchor strength: 0.25
PromptRelay epsilon:       0.3
```

PromptRelay `segment_lengths` is left empty in this example. Its three `|`-separated local prompt segments are therefore distributed evenly across the clip, matching the Time-Gated LoRA `middle third` region.

### Reproducing the Claymation example

1. Copy `examples/inputs/claymation_demo_input.jpg` to your ComfyUI `input/` directory.
2. Load `workflows/PromptRelay_LTXV_TimeGatedLoRA_Claymation_v1.0rc1.json`.
3. Install or supply the model and LoRA dependencies listed below.
4. Adjust local model filenames if your ComfyUI installation uses different folder names.

## Demo media

* [Age Slider Reveal — directional transformation demo](examples/videos/Age_Slider_Reveal_Licon_Enabled.mp4)
* [Claymation Style Reveal — realistic → claymation → realistic](examples/videos/Claymation_Style_Reveal_Licon_Enabled.mp4)

The bundled demo video copies have been remuxed without user or workflow metadata.

## Demo dependencies and credits

Third-party LoRAs used in demo workflows or videos are **not included** in this repository and are not required to use the custom node itself.

| Demo asset / LoRA                                                                      | Role in the demonstration                                                                 | Credit / source status                                                                                                          |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Claymation LoRA (`LTX/Claymation.safetensors` in the example workflow)                 | Time-gated style LoRA active in the middle third                                          | [vrgamedevgirl84 — LTX 2.3 Clay Mation Style LoRA](https://huggingface.co/vrgamedevgirl84/LTX_2.3_Clay_Mation_Style_LoRa)       |
| Licon VBVR I2V LoRA (`LTX/Ltx2.3-Licon-VBVR-I2V-390K-R32.safetensors`, strength `1.0`) | Quality/consistency LoRA retained because disabling it substantially reduced demo quality | [LiconStudio — Ltx2.3-VBVR-lora-I2V](https://huggingface.co/LiconStudio/Ltx2.3-VBVR-lora-I2V)                                   |
| Age Slider LoRA (`age_slider_ltx23_v_0_5.safetensors`)                                 | Directional age-transformation demo video                                                 | Third-party LoRA; original source and author could not currently be verified. Weights are not distributed with this repository. |


