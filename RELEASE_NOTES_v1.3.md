# Release Notes — v1.3.0

## Temporal Director arrives

Version 1.3.0 is the largest update to **ComfyUI-LTXV-TimeGated-LoRA** so far.

The original Time-Gated LoRA node remains available, but this release adds a new coordinated workflow for LTX 2.3: **Temporal Director**.

Temporal Director gives prompt changes and one or more LoRAs a shared timeline. Instead of scheduling each part of the workflow independently, the prompt relay, LoRA envelopes, segment boundaries, and visual preview all use the same canonical `segment_data`.

The result is a workflow that is easier to reason about, easier to debug, and much easier to tune visually.

---

## Highlights

- Shared schedule for up to **4 prompt segments**
- Scheduled Prompt Relay integration
- Stackable Scheduled Time-Gated LoRA nodes
- Seconds-first visual schedule preview
- Independent incoming and outgoing q-curves
- Movable curve anchors for leads, delays, overlaps, and centered peaks
- Symmetric first- and last-segment behavior
- Automatic recovery of missing prompt separators
- Native sigma tail and trim utility
- Full retention of the classic Time-Gated LoRA workflow

---

## New Temporal Director nodes

### LTXV Schedule Sync

Creates the canonical shared schedule used by the complete Director workflow.

It resolves:

- total rendered frames
- latent frame count
- FPS and duration
- segment count
- equal or manual boundaries
- segment lengths
- cleaned local prompts
- parser warnings and normalization notes

The resulting `segment_data` can be shared by Prompt Relay, every Scheduled TimeGate, and the visual preview.

### LTXV Prompt Relay Encode (Scheduled)

Encodes a sequence of local prompts against the shared schedule without exposing a separate timeline UI.

It also emits `relay_timing`, allowing Scheduled TimeGate nodes to derive their transition softness from the same Prompt Relay epsilon value.

This keeps prompt routing and LoRA timing coupled while preserving independent control over the position and shape of each LoRA envelope.

### LTXV Time-Gated LoRA (Scheduled)

Applies a visual LTX 2.3 LoRA to one selected schedule segment.

Multiple Scheduled TimeGate nodes can be chained and can target different segments while remaining synchronized to the same prompt timeline.

Each node supports:

- `strength_before`
- `strength_during`
- `strength_after`
- `flat` or `q_curve` ramps
- independent `q_in` and `q_out`
- independent `q_in_shift` and `q_out_shift`
- stackable runtime-safe model patching
- reusable data output for the schedule preview

### LTXV Temporal Schedule Preview

Displays the complete schedule before sampling.

The preview includes:

- total duration in seconds
- rendered and latent frame counts
- FPS
- segment boundaries
- per-segment durations
- prompt excerpts
- up to four LoRA envelopes
- incoming and outgoing transition markers
- parser warnings and automatic normalizations

Seconds are shown prominently, while frame counts remain available for technical reference.

---

## Flexible LoRA envelopes

Version 1.3.0 separates curve **shape** from curve **position**.

### Curve shape

`q_in` and `q_out` control how sharply or gradually the LoRA enters and leaves.

### Curve position

`q_in_shift` and `q_out_shift` move the transition anchors across neighboring segment regions.

This allows effects such as:

- a LoRA beginning before the corresponding prompt change
- a prompt change leading while the LoRA follows later
- a LoRA holding into the next prompt segment
- an early fade-out before the target segment ends
- a centered peak with no plateau
- a rise and fall spanning several semantic segments

At their extreme positions, the shift controls can reach the midpoint of the neighboring or target segment, depending on direction.

---

## Symmetric edge behavior

The first and last video segments no longer behave as special cases.

Temporal Director creates virtual pre-roll and post-roll regions with the same geometry as real neighboring segments. Curves are calculated normally and then clipped to the visible video range.

This preserves the same transition behavior at every boundary:

- `strength_before` remains meaningful for the first segment
- `strength_after` remains meaningful for the last segment
- incoming and outgoing shifts remain functional at both video edges
- off-screen curve portions are not compressed into the visible range

The same envelope logic therefore applies to first, middle, last, and single-segment schedules.

---

## Automatic prompt separator recovery

LLM-generated prompt sequences do not always contain reliable `|` separators.

Schedule Sync can now recover segment boundaries automatically.

### Explicit separators

```text
Prompt one | Prompt two | Prompt three
```

### Ordered labels

```text
segment 1: Prompt one

segment 2: Prompt two

segment 3: Prompt three
```

Labels such as `segment`, `scene`, or `shot` are recognized when they form an ordered sequence. The labels are removed and the missing separators are inserted automatically.

### Blank-line paragraphs

When no pipes or ordered labels are present, 2–4 blank-line-separated paragraphs can be interpreted as individual segments.

Existing pipes always take precedence.

The preview reports every normalization so repaired prompts remain transparent and inspectable.

---

## Seconds-first schedule preview

The Temporal Schedule Preview now presents time in the form most users perceive it:

```text
12.04 s · 289 frames · 37 latent · 24.00 fps · 3 segments
```

Each segment is also labeled by duration first:

```text
[1] 0.00–4.00 s
96 frames
```

The maximum prompt excerpt length can be increased up to **640 characters**, allowing the preview to show substantially more of each segment prompt when sufficient space is available.

---

## Native sigma tools

### LTXV Sigma Tail / Trim

A new utility node for editing native ComfyUI `SIGMAS` tensors without converting them to float lists.

Supported operations include:

- drop the first N sigma values
- keep the final N sigma values
- keep a final fraction of the schedule
- start at or below a selected sigma threshold
- ensure a zero endpoint

This is useful for controlled second-pass and refinement workflows where the high-noise portion of a scheduler should be removed while retaining the scheduler's native remaining sequence.

---

## Classic Time-Gated LoRA remains available

The established classic node is still included:

- **LTXV Time-Gated LoRA (LTX 2.3)**
- **LTXV Envelope Curve Preview / Analyze**
- **LTXV Temporal Envelope Inspector**
- **LTXV Semantic Segment Planner**

Existing workflow concepts and historical envelope modes remain supported.

Temporal Director is an additional workflow style, not a replacement for the original node.

---

## Typical Temporal Director graph

```text
LTXV Schedule Sync
 ├─ segment_data ──> Prompt Relay Encode (Scheduled)
 ├─ segment_data ──> Time-Gated LoRA (Scheduled) #1
 ├─ segment_data ──> Time-Gated LoRA (Scheduled) #2
 └─ segment_data ──> Temporal Schedule Preview

Prompt Relay Encode (Scheduled)
 └─ relay_timing ──> all Scheduled TimeGate nodes

Scheduled TimeGate #1
 └─ data ──> Preview data_1

Scheduled TimeGate #2
 └─ data ──> Preview data_2
```

The same `video_latent` should be supplied to Schedule Sync and all Scheduled TimeGate nodes so every component resolves the same rendered timeline.

---

## Upgrade notes

- Package version is now `1.3.0`
- The classic node remains available under its established identity
- Scheduled nodes added during development should be freshly inserted into older test workflows if their widget layout no longer matches
- Existing v1.1 workflow files are not replaced
- No additional Python dependencies are required

For a clean upgrade:

1. update or replace the custom-node directory
2. restart ComfyUI completely
3. hard-refresh the browser
4. reload the workflow
5. replace any Scheduled node that was saved with an early development build and shows shifted widgets

---

## Scope and current limitations

- Temporal gating currently targets **visual LTX 2.3 LoRA layers**
- Temporal audio LoRA gating is not included in v1.3.0
- Temporal Director supports up to **4 semantic prompt segments**
- Prompt Relay epsilon controls transition softness, while TimeGate shift controls move the transition anchors
- LoRA compatibility and visual quality still depend on the LoRA, checkpoint, prompt, sampler, and workflow
- The schedule preview visualizes the configured envelope; it does not predict whether a LoRA itself will introduce ghosting, morphing, or other generation artifacts

Testing unfamiliar LoRAs individually before stacking them remains strongly recommended.

---

## Validation

The v1.3.0 release candidate was exercised across numerous LTX 2.3 I2V workflows, including long and short clips, one- and two-pass generation, and multiple stacked Scheduled TimeGate nodes.

The tested areas include:

- flat schedules
- q-curve schedules
- positive and negative q shifts
- centered peaks
- first-segment pre-roll
- last-segment post-roll
- non-zero `strength_before` and `strength_after`
- multiple stacked LoRAs
- equal segment boundaries
- automatic separator recovery
- seconds-first preview rendering
- prompt excerpts up to 640 characters
- native sigma trimming
- Python compilation
- JavaScript syntax validation

As always, generated-video behavior varies with model and LoRA content, so visual output should be evaluated independently from scheduler correctness.

---

## Prompt Relay attribution

Prompt scheduling support references the Prompt Relay method by:

- Gordon Chen
- Ziqi Huang
- Ziwei Liu

Project page:

- https://gordonchen19.github.io/Prompt-Relay/

ComfyUI implementation reference:

- https://github.com/kijai/ComfyUI-PromptRelay

See [NOTICE.md](NOTICE.md) for attribution details.

---

## Thank you

Version 1.3.0 grew from practical workflow testing rather than from an abstract scheduler design.

The central goal remains simple:

> Make LoRA timing visible, predictable, and directly controllable across an LTX 2.3 video.

Feedback, reproducible test cases, and real-world workflow reports are welcome.
