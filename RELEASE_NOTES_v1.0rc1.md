# v1.0rc1 — Release Candidate

## Highlights

- Temporally gate visual LTX 2.3 LoRAs by halves, thirds, quarters, or manual frame schedules.
- Stack multiple independent time-gated LoRA nodes in one continuous LTX 2.3 sampling run.
- Required `video_latent` timing reference with passthrough output for clean stacking.
- PromptRelay-compatible workflow usage documented.
- Strength controls use practical 0.05 increments.

## Validated in this release candidate

- PromptRelay I2V workflow with a Claymation-style LoRA active only in the middle third.
- Return to realistic appearance after the active style region.
- `video_latent` passthrough.
- UI tooltips and inactive-field greying.

## Scope and known limitations

- Visual LTX 2.3 LoRA gating only; temporal audio LoRA gating is not supported.
- `torch.compile` / `TorchCompileModelAdvanced` is not supported.
- Real LoRA effects may semantically influence later frames even after their gated region ends.

## Draft-release note

This is a local/private preview draft. The repository is prepared with an MIT license attributed to `Jinx138`; third-party demo LoRA credits must still be verified before public publication.


## Bundled local preview assets

- Validated Claymation PromptRelay example workflow.
- Metadata-clean demo input image for the Claymation workflow.
- Claymation style reveal demo video.
- Age-slider reveal demo video.

Third-party demo LoRAs are not included and must be sourced separately.
