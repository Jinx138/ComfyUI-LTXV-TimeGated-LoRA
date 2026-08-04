# Notices and Attribution

This package includes the LTXV Temporal Director integration.

Prompt Relay attribution:
- Original method: Gordon Chen, Ziqi Huang, Ziwei Liu, "Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation".
- ComfyUI adaptation reference: kijai/ComfyUI-PromptRelay.

The Scheduled adapter does not vendor Kijai's full PromptRelay node UI. It locates an installed ComfyUI-PromptRelay extension at runtime and adapts its helper modules (`prompt_relay.py`, `patches.py`) to LTXV Schedule Sync segment data.
