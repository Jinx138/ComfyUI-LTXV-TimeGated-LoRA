# ComfyUI-LTXV-TimeGated-LoRA v1.3.0

## Temporal Director

The Temporal Director branch is intentionally limited to **max 4 prompt segments / 3 boundaries**. `LTXV Schedule Sync` tolerates missing `|` separators by reconstructing them from ordered `segment/scene/shot` labels or, as a fallback, 2-4 blank-line paragraphs. Scheduled q-curves use a half-cosine shoulder across segment boundaries. PromptRelay `epsilon` defines the coupled zero-position anchor, while `q_in_shift` / `q_out_shift` move the transition phase within the same segment timeline. The inward extrema (`q_in_shift=+1`, `q_out_shift=-1`) meet at the target midpoint and can form a centered LoRA peak. First and last targets use full, equally long virtual neighbor segments, so their curves behave exactly like internal curves and simply continue outside the visible clip. The preview presents seconds first and shows seconds for every segment and phase shift. Use `flat` for hard segment edges.

> **v1.3.0 release:** keeps the established `LTXV Time-Gated LoRA (LTX 2.3)` node and adds the shared-schedule **Temporal Director** workflow beside it.

## New Temporal Director nodes

The Director branch makes one upstream schedule the single source of truth for PromptRelay and TimeGate timing.

```text
LTXV Schedule Sync
    ├─ segment_data -> LTXV Prompt Relay Encode (Scheduled)
    ├─ segment_data -> LTXV Time-Gated LoRA (Scheduled)
    └─ segment_data -> LTXV Temporal Schedule Preview
```

### LTXV Schedule Sync

- Takes the final video-only `video_latent`, `fps`, and pipe-separated `local_prompts`.
- Prefers explicit `|` separators, but automatically reconstructs missing separators from ordered labels such as `segment 1:` / `scene 2:` / `shot 3:`.
- If no labels are available, 2-4 blank-line paragraphs can be used as a fallback segment structure.
- Emits exact visible-frame segment lengths such as `168,168,169`.
- Supports `expected_segments` plus `warn/error` checks after recovery; repaired prompts are reported as normalizations rather than errors.
- No independent manual-frame timeline; optional normalized boundary sliders/bar remain part of the shared Schedule Sync plan.

### LTXV Prompt Relay Encode (Scheduled)

- Adapter around Kijai PromptRelay.
- No timeline UI and no manual segment length field.
- Uses `segment_data.segment_texts` and `segment_data.segment_lengths_csv` from Schedule Sync.
- Optional `RELAY_OPTIONS` input accepts Kijai's Prompt Relay Advanced Options.
- Requires ComfyUI-PromptRelay to be installed/enabled.

### LTXV Time-Gated LoRA

- New scheduled TimeGate variant without `manual_frames` or `effect_region` presets.
- Uses `target_segment` from the shared Schedule Sync segment plan.
- `q_in` and `q_out` shape the incoming and outgoing neighboring-segment ramps.
- `q_in_shift` / `q_out_shift` retain the range `-1.0..+1.0`, but now use segment-relative phase travel rather than one fixed softness width.
- `q_in_shift=+1` completes the incoming ramp at the target midpoint; `q_out_shift=-1` starts the outgoing ramp there. Together they can produce a centered peak with little or no plateau.
- The outward extrema reach the midpoint of the preceding/following segment; first/last targets use equally long virtual pre-roll/post-roll segments.
- The envelope is assembled as `before -> incoming ramp -> during -> outgoing ramp -> after`, so `strength_during` remains active until the shifted outgoing transition actually starts.
- Internal-segment `0.0` behavior remains unchanged; edge segments intentionally use the same full virtual-neighbor geometry as internal segments.
- Intended to sit after PromptRelay Scheduled and before the sampler/guider.

### LTXV Temporal Schedule Preview

- Replaces the curve-only preview for Director workflows.
- Shows total duration and segment time ranges in seconds first, with frame counts retained as secondary technical information.
- Shows segment texts, Schedule Sync warnings, and up to four TimeGate curves. `max_text_chars` now accepts up to 640 characters per segment; visible lines are limited only by the selected preview height and box width.
- Non-zero incoming/outgoing phase shifts appear as dashed `IN` / `OUT` markers and are reported in seconds and frames.

## Director workflow sketch

```text
video_latent -> LTXV Schedule Sync

MODEL -> LTXV Prompt Relay Encode (Scheduled)
      -> LTXV Time-Gated LoRA (Scheduled) # target segment 1/2/3...
      -> optional additional scheduled TimeGates
      -> optional global/distill LoRA
      -> sampler

segment_data + TimeGate data outputs -> LTXV Temporal Schedule Preview -> Preview Image
```

The old nodes remain available for compatibility:

- `LTXV Time-Gated LoRA (LTX 2.3)`
- `LTXV Semantic Segment Planner`
- `LTXV Envelope Curve Preview`
- `LTXV Temporal Envelope Inspector`

---

## Legacy Time-Gated LoRA node


Node: **LTXV Time-Gated LoRA (LTX 2.3)**

Temporally apply visual LTX 2.3 LoRAs to selected regions of a single continuous video sampling run. Use it for effects such as a style appearing only in the middle third, a transformation beginning in the final quarter, or multiple independent scheduled LoRAs chained in one clip.

## Scope

- Visual LTX 2.3 LoRA layers only; temporal audio gating is intentionally not supported in the current release.
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
- `q_in`, `q_out`: separate incoming/outgoing q-curve controls. Legacy `ramp_q` workflows are represented by both values using the old q value.
- `transition_frames`: crossfade width at effect boundaries; defaults to `0` and is only used by manual/legacy transition modes.
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
- Temporal audio LoRA gating is not supported in the current release.
- A real LoRA may influence later video semantics after its active region even when the gate itself is temporally correct.

## Validated demo scenarios

### Age Transformation Reveal

A single continuous I2V run showing an adult appearance, a younger reveal and an elderly reveal using a time-gated age/slider LoRA with three PromptRelay segments. A reduced I2V image anchor was critical for visible transformation.

### Claymation Style Reveal

A realistic scene reveals a Claymation style only in the middle third, then returns to realism. The sign's blank reverse side acts as a natural occluder/reveal transition. This workflow is the recommended release-candidate smoke test.


## Envelope diagnostics / curve preview

The legacy diagnostics path contains three nodes:

- **LTXV Time-Gated LoRA (LTX 2.3)** — productive model patching node.
- **LTXV Temporal Envelope Inspector** — non-patching audit node. It resolves the frame profile, latent profile, regions, samples and warnings.
- **LTXV Envelope Curve Preview (CPU)** — renders a small timeline/strength analyzer image from the Inspector's combined `data` output.

Recommended diagnostic wiring:

```text
video_latent
  → LTXV Temporal Envelope Inspector
      data → LTXV Envelope Curve Preview (CPU) → Preview Image
      report_md → Show Text / PreviewAny
```

The `data` output is a single custom payload (`LTXV_ENVELOPE_DATA`) so the preview node does not need separate cables for timeline, profile, regions and warnings. Rendering is CPU-side via PIL/numpy and returns a normal ComfyUI `IMAGE`.

## Installation

Extract or clone the repository into ComfyUI `custom_nodes`, restart ComfyUI, and hard-refresh the browser. Workflows created with development builds should use freshly added Scheduled nodes because their widget layout changed during the v1.3 development cycle.

## v1.1 visual envelope preview

v1.1 adds a CPU-rendered curve preview that shows the actual local LoRA envelope over the full video timeline.

The preview is useful for checking selected regions, comparing stacked Time-Gated LoRA nodes, debugging `flat` versus `q_curve`, and verifying `data` / `data_2` overlays before sampling.

### Example: late q-curve gate

![Late q-curve gate](examples/images/curve_q_late_gate.png)

A delayed q-curve where the LoRA stays inactive for most of the clip and ramps strongly into the last third.

### Example: three-step flat envelope

![Rain step gate](examples/images/curve_flat_step_gate.png)

A flat local envelope using one node as a 0 → 0.5 → 1 step controller.

### Example: stacked Time-Gated LoRA nodes

![Stacked wiring example](examples/images/workflow_stacked_wiring.png)

Two Time-Gated LoRA nodes can be stacked in the MODEL path. Their `data` outputs can be connected to the Curve Preview / Analyze node as `data` and `data_2` to compare both envelopes in one plot.

### Comparison videos

The repository also includes two three-up comparison videos:

- [Cucumber comparison — No LoRA / Standard LoRA / Time-gated LoRA](examples/videos/threeup_lora_comparison_cucumber.mp4)
- [Rain comparison — No LoRA / Standard LoRA / Time-gated LoRA](examples/videos/threeup_lora_comparison_rain_repo.mp4)

The cucumber comparison is a deliberately playful A/B showcase. The rain comparison is useful as a production/stress-test example: it demonstrates a rain/particle LoRA and also shows why stochastic, high-frequency detail LoRAs can be more sensitive to temporal gating than semantic or stylistic transformations.

### Production workflow example

The repository includes a production/testing workflow:

`workflows/production/PromptRelay_LTXV_TimeGated-LoRA_Rain_StepGate_v1.1.json`

This is not a minimal install test. Bypassed nodes are intentionally kept for A/B testing, alternate gate setups, preview inspection, last-frame export, upscaling, and production debugging. Input images and third-party LoRAs are not included.

### Note on stochastic / particle LoRAs

Time-gated curves are especially useful for semantic or stylistic transformations such as realism ↔ claymation, age sliders, mood/look changes, or character/style intensity. Fine stochastic detail LoRAs such as rain, snow, dust, film grain, water ripples, or particle effects may flicker under gradual curves because the model keeps renegotiating high-frequency details over time. For these LoRAs, test lower `lora_strength`, `flat` instead of a long `q_curve`, a non-zero `strength_before`, or constant full-clip application.

## Sigma Tail utility

`LTXV Sigma Tail / Trim` accepts native ComfyUI `SIGMAS` and outputs native `SIGMAS` again. It is intended for later LTX passes where the highest-noise portion of an existing scheduler curve should be skipped without converting sigmas to ordinary float lists.

Supported modes include:

- dropping a fixed number of leading sigma values;
- keeping a fixed number or fraction of the tail;
- starting at the first sigma at or below a chosen threshold;
- optionally ensuring a final zero sigma.

Example:

```text
BasicScheduler(linear_quadratic, 24 steps)
→ LTXV Sigma Tail / Trim(start_at_or_below_sigma=0.80)
→ sampler
```

## Dependencies and attribution

The Scheduled PromptRelay adapter requires an installed and enabled `ComfyUI-PromptRelay`. It does not vendor the PromptRelay UI; see [NOTICE.md](NOTICE.md) for attribution and integration details.

## Release history

See [CHANGELOG.md](CHANGELOG.md) and [RELEASE_NOTES_v1.3.md](RELEASE_NOTES_v1.3.md).
