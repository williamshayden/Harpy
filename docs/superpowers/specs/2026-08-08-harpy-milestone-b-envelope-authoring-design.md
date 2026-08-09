# Harpy Milestone B Envelope Authoring and Frequency Control Design

Date: 2026-08-08

Status: approved foundation; graph-native editor revision awaiting written review

Revision: 2026-08-08. The graph-native editor decisions below supersede the original
persistent ADSR/curvature field grid and curve-only reset. The DSP, patch, deferred-apply,
codec, frequency-control, and Milestone C boundaries are unchanged.

## Intent

Milestone B turns Harpy's immutable sine patch into a precisely editable native
instrument while preserving the deterministic research API established by Milestone A.
It adds graph-native ADSR authoring, one constrained quadratic Bézier curve for each
shaped stage, strict patch-schema version 2, a compact envelope inspector, and a
professional frequency-knob redesign.

Milestone B is not the reinforcement-learning milestone. The sine-only Gym remains
Milestone C because its actions, observations, rewards, leakage controls, model adapters,
and benchmark protocol need a separate design pass.

This specification supersedes only the Milestone A statements that explicitly deferred
editable envelopes and knob polish. The Milestone A architecture, audio safety,
measurement semantics, and native-only product boundary remain authoritative unless this
document changes them explicitly.

## Approved product decisions

The following decisions are locked:

1. Milestone B includes ADSR/curve authoring and frequency-knob refinement.
2. Milestone C remains the sine-only Gym proof.
3. Envelope edits do not automate the currently sounding voice. A held or releasing
   voice finishes with the patch on which it started; the newest valid authored patch
   is applied once the voice is truly idle and is heard on the next Play.
4. Attack, Decay, and Release each use one constrained quadratic Bézier control. Sustain
   remains a flat level.
5. The graph and DSP share one pure mathematical implementation. Painted geometry is
   never a second envelope representation.
6. Patch JSON version 1 remains readable. Harpy writes strict canonical version 2.
7. The existing continuous logarithmic C2-C4 tuning behavior remains unchanged. The
   knob work is a professional visual and interaction-discoverability redesign.
8. Harpy remains a native PySide6 desktop application. No browser surface is added.
9. Envelope values are manipulated in context on the graph. Permanent A/D/S/R and
   curvature text-box rows are not part of the final surface.
10. The contextual `Reset` action restores the complete envelope default—A/D/S/R and
    all three curves—without changing pitch, oscillator facts, output gain, or any
    authored patch field outside the envelope. Normal measurement invalidation and
    deferred application still apply to the resulting patch edit.

## Goals

Milestone B must provide:

- direct scrubbing and transient exact editing of Attack, Decay, Sustain, and Release
  on the envelope graph;
- one graphical curvature handle for Attack, Decay, and Release;
- deterministic monotone envelope rendering with exact endpoints;
- sample-for-sample compatibility for the zero-curvature Milestone A envelope;
- immutable complete-patch commits rather than per-parameter audio commands;
- deferred, coalesced patch application while a voice is active or releasing;
- strict v1-to-v2 patch migration and canonical atomic v2 saves;
- a compact native envelope inspector that does not reduce either plot's data canvas
  below the Milestone A minimum;
- a frequency control that reads as a credible pro-audio dial and communicates its
  interaction model;
- keyboard and assistive-technology access for every new editor control;
- automated and native acceptance evidence suitable for a public research repository.

## Non-goals

Milestone B does not add:

- live envelope automation or audible parameter changes during a held note;
- command coalescing for sample-accurate modulation;
- cubic or free two-dimensional Bézier handles;
- arbitrary multi-stage envelopes, sustain loops, delay, hold, or modulation routing;
- oscillator, output-gain, filter, wavetable, or effect editing;
- additional waveforms, polyphony, velocity, MIDI input, or imported-audio editing;
- preset libraries, recent files, autosave, undo history, or a database;
- Gymnasium, actor-facing observations, rewards, model adapters, training, or benchmark
  reporting;
- JUCE, C++, a browser UI, or a second audio backend.

## Foundation retained from Milestone A

The following boundaries remain unchanged:

- `SynthPatch` is immutable, serializable sound configuration. It contains no selected
  note, frequency, gate, capture, file path, window geometry, or device state.
- `RenderConfig` remains the machine/render truth.
- `SynthEngine` remains the authoritative deterministic mono NumPy renderer.
- Frequency is supplied directly in hertz and is independent of patch configuration.
- The engine owns oscillator phase, envelope performance state, and the active voice.
- The controller sends semantic commands across the Qt callback boundary; the GUI never
  mutates the audio-thread engine.
- Patch replacement resets performance state, discards staged old PCM, and begins with
  exact silence until a later Play.
- Capture generations prevent a pre-replacement analysis result from restoring stale
  data.
- The pure `AudioObservation` API remains shared by the GUI and future research clients.
- Qt remains a presentation and device-edge dependency, not a core synthesis dependency.

## Architecture

Milestone B adds four bounded responsibilities:

1. **Curve mathematics** — a pure core function evaluates constrained quadratic
   segments and samples previews.
2. **Envelope renderer** — the current private linear envelope becomes a generic ADSR
   state machine that consumes the immutable curve-enabled configuration.
3. **Authored-patch lifecycle** — the workbench controller distinguishes the latest
   valid authored patch from whether that patch has reached the audio engine.
4. **Native authoring surface** — a dedicated envelope editor owns draft widgets and
   emits complete immutable patch candidates.

The dependency direction is:

```text
EnvelopeConfig ──> pure quadratic segment ──> ADSR renderer ──> SynthEngine
       │                    │
       │                    └───────────────> envelope preview sampler
       │                                             │
       └──> strict patch codec                       └──> native EnvelopeEditor
                                                          │
                                                          v
                                                 WorkbenchController
                                                          │
                                                          v
                                                    Qt audio adapter
```

The preview imports pure core mathematics. The core never imports Qt, pyqtgraph, or the
editor.

## Curve-enabled patch model

`EnvelopeConfig` becomes:

```python
EnvelopeConfig(
    attack_seconds: float,
    decay_seconds: float,
    sustain_db: float,
    release_seconds: float,
    attack_curve: float,
    decay_curve: float,
    release_curve: float,
)
```

The default remains:

```text
attack_seconds = 0.001
decay_seconds  = 0.600
sustain_db     = -6.0
release_seconds = 0.600
attack_curve   = 0.0
decay_curve    = 0.0
release_curve  = 0.0
```

Validation is total and field-specific:

- Attack, Decay, and Release are finite, strictly positive, and must produce at least
  one frame under the composed render configuration.
- Duration-to-frame multiplication must remain finite; pathological finite inputs are
  rejected as named `ValueError`s rather than leaking `OverflowError`.
- Sustain is finite and no greater than `0 dB`.
- Every curvature is a JSON/Python number, never a boolean, finite, and within the
  inclusive interval `[-1.0, +1.0]`.
- `0.0` is exactly the Milestone A linear curve.

The model does not impose an arbitrary maximum envelope duration. The native editor uses
validated text entry rather than a duration slider, and the authoring graph compresses
time for display.

## Constrained quadratic Bézier contract

For a stage beginning at amplitude `a`, ending at amplitude `b`, and carrying curvature
`c` in `[-1, +1]`, the control point is:

```text
P0 = (0.0, a)
P1 = (0.5, (a + b) / 2 + c * (b - a) / 2)
P2 = (1.0, b)
```

Because the control point's x-coordinate is `0.5`, `x(t) = t`. The amplitude is:

```text
B(a, b, c, t) = (1 - t)^2 * a
               + 2 * (1 - t) * t * P1.y
               + t^2 * b
```

where `t` is constrained to `[0, 1]`.

This family has the required properties:

- `B(..., 0) = a` and `B(..., 1) = b` exactly;
- `c = 0` is a straight line;
- `c = -1` leaves the starting level gently and finishes faster;
- `c = +1` leaves the starting level faster and approaches the target gently;
- the control level remains between the endpoints, guaranteeing monotonic output and no
  overshoot;
- the sign semantics are identical for rising and falling segments: positive curvature
  reaches toward the target earlier;
- the function is independent of block size and GUI geometry.

Attack evaluates `0 -> 1`. Decay evaluates `1 -> sustain_amplitude`. Release evaluates
the actual last-emitted level at note-off `-> 0`. Sustain is a constant level and has no
curvature.

The pure API owns both scalar evaluation and preview sampling. It validates all inputs,
returns core numerical arrays, and has no performance state.

## Discrete-time ADSR semantics

Milestone A's sample-domain transition convention is retained:

- A segment containing `F` frames emits normalized positions `1/F, 2/F, ..., F/F`.
- The first emitted sample therefore moves away from the segment's starting level.
- The `F`th sample is the exact target and the next stage becomes active after that
  sample is emitted.
- Note-on resets the envelope to zero and starts Attack.
- Attack's final sample is exactly `1.0`.
- Decay's final sample is exactly `sustain_amplitude`.
- Sustain emits that exact level until note-off.
- Note-off during Attack, Decay, or Sustain snapshots the last emitted level and begins
  a full-duration Release from that level.
- Repeated note-off during Release is idempotent.
- Release's final sample is exact zero; the next render is exact silence and the voice
  becomes idle.
- Monophonic retrigger resets oscillator phase and restarts Attack from zero, matching
  Milestone A.

Zero curvature must preserve Milestone A's characterized linear sample sequence exactly.
The implementation may retain a linear fast path or use a recurrence whose zero second
difference reproduces the existing additions; this is a compatibility requirement, not
an implementation preference.

The generic renderer remains private to the synth package. It replaces the misleading
`LinearEnvelope` name rather than preserving a compatibility alias that suggests only
linear behavior.

## Patch JSON schema version 2

Version 2 retains the root structure and replaces the v1 envelope curve string:

```json
{
  "schema_version": 2,
  "oscillator": {
    "type": "sine"
  },
  "envelope": {
    "attack_seconds": 0.001,
    "decay_seconds": 0.6,
    "sustain_db": -6.0,
    "release_seconds": 0.6,
    "attack_curve": 0.0,
    "decay_curve": 0.0,
    "release_curve": 0.0
  },
  "output_gain_dbfs": -12.0
}
```

Codec requirements:

- Version 1 decoding remains strict and unchanged. Its envelope requires exactly the v1
  keys and `"curve": "linear_amplitude"`.
- A valid v1 document maps all three runtime curvature values to `0.0`.
- Version 2 requires exactly the shown v2 keys. It does not accept the old `curve` key.
- Unknown versions fail explicitly.
- Missing, extra, duplicate, incorrectly typed, boolean, non-finite, and out-of-range
  values fail with the most specific available field path.
- The decoder constructs and validates a complete immutable patch before returning it.
- Runtime `SynthPatch` does not store a schema-version field.
- Saving always emits canonical schema version 2 in the shown key order, with two-space
  indentation, UTF-8, shortest round-trippable finite numbers, and one trailing newline.
- Saving continues to use a fully written and closed temporary sibling followed by
  atomic replacement. Failed overwrites preserve the previous file.

Load and Save As remain one-off native operations. Harpy does not remember paths, create
recent-file state, or add a preset browser.

## Authored, pending, and applied patch lifecycle

The latest valid user intent and the audio engine's applied patch can temporarily differ
while a voice finishes. The state is explicit rather than inferred from widgets.

`WorkbenchState.patch` remains the latest valid authored patch so existing consumers and
Save As retain one clear configuration truth. The controller adds:

```text
patch_apply_state: applied | pending
```

The audio adapter continues to own the actual engine instance. The controller does not
read engine internals. Here, `applied` means the replacement has been admitted to the
serialized audio-command path and any later `NOTE_ON` is ordered after it; it does not
claim that the callback has already rendered a block with the patch.

For deferred application, `voice_idle` has a stricter boundary than renderer state
alone. When a rendered block observes the transition to envelope-idle, the audio source
records the end of that encoded release-bearing block in its replaceable staging buffer.
It emits the generation-tagged idle notification only after bytes through that watermark
have been handed to the audio sink. PCM already accepted by the sink is not replaceable.
A reset, file load, device failure, or shutdown remains an explicit force-stop and may
discard staged PCM; an ordinary deferred editor commit may not. This makes “truly idle”
mean both envelope-idle and release-tail-drained at the source boundary. The conservative
block watermark may defer application by less than or equal to one render block of
trailing silence; it cannot shorten the release.

### Editor commit while idle

1. The editor validates a complete candidate patch.
2. The controller records it as `authored_patch`.
3. Capture is reset into a new generation.
4. One `REPLACE_PATCH` command crosses the audio boundary.
5. The state is `applied`; the next Play uses the new patch.

### Editor commit while a voice is held or releasing

1. The editor validates a complete candidate patch.
2. The current voice continues unchanged.
3. The controller records the candidate as the latest `authored_patch` and marks it
   `pending` without sending a patch command.
4. Additional valid edits replace the pending candidate in memory; they do not create a
   command stream.
5. Once the generation-tagged idle notification for the current voice is accepted, the
   controller clears capture and sends exactly one `REPLACE_PATCH` containing the latest
   candidate.
6. The state becomes `applied`, and the next Play is enabled.

While a patch becomes pending during a held gate, the active Play gesture remains able
to deliver its matching release; disabling the pressed widget must not strand the gate.
After note-off is dispatched, Play is disabled until the pending replacement is admitted
to the command path. Independently of widget state, the controller rejects a new
`NOTE_ON` while `patch_apply_state` is `pending`, so the old patch cannot be retriggered
ahead of replacement. The current release remains audible and retains its measurement
until the pending patch is applied, at which point capture clears.

An invalid or syntactically incomplete editor draft never replaces the latest valid
authored patch, never changes pending state, and never emits a command. Escape restores
the latest valid authored values.

Save As writes the latest valid authored patch, including a pending one. The inspector
labels that condition so the user is not told that an unauditioned patch is already
applied.

Patch-file Load retains its explicit Milestone A behavior rather than becoming a deferred
editor gesture: it force-stops playback, validates and replaces atomically, clears
capture, discards any local invalid draft, and leaves the loaded patch applied.

## Device failure, recovery, and shutdown

- Editing remains available while audio is unavailable.
- With no active voice, a valid edit is sent through the existing backend cache path and
  becomes the patch used on device recovery.
- If device failure force-stops a voice while an authored patch is pending, the
  generation-tagged `REPLACE_PATCH` is the force-stop/reset operation and installs the
  latest valid authored patch into the unavailable backend cache. A separate `RESET`
  command is not emitted.
- Recovery creates an idle engine from the cached latest patch, clears the error, and
  enables Play without healthy-status copy.
- Shutdown remains idempotent. It does not autosave an authored or invalid draft.
- A stale pre-failure idle callback cannot apply or clear a newer pending patch.

## Native envelope editor

The authoring surface is a dedicated native component rather than more logic added to
`HarpyWindow`.

Responsibilities:

- `EnvelopeEditor` owns the authored patch, local draft, validation feedback, and the
  sole complete-patch commit path.
- `EnvelopeGraph` paints the preview and owns direct-manipulation interactions. It emits
  proposed envelope-field values; it does not own patch truth.
- A pure preview sampler produces normalized stage curves from `EnvelopeConfig`.
- `HarpyWindow` composes the editor and routes one complete candidate to the controller.

The editor displays:

- one inline value control for each stage: `A 1 ms`, `D 600 ms`, `S -6 dB`, and
  `R 600 ms` for the default envelope;
- one control handle for Attack, Decay, and Release;
- a transient precise curvature readout while a handle is hovered, focused, or dragged;
- a contextual `Reset` action in the Envelope inspector;
- patch status: `Active`, `Editing`, or `Pending`;
- compact Sine and output-gain facts;
- Load and Save As actions.

The inline stage controls are display-first rather than permanent text boxes. Press-drag
vertically on A, D, or R to make a relative multiplicative duration adjustment; press-drag
vertically on S to make a linear decibel adjustment. Every drag is anchored to the value
at pointer press, so graph relayout cannot feed back into the authored value. Moving up
increases the value and moving down decreases it. A drag previews the local draft and
attempts exactly one complete-patch commit on release.

The interaction scale is deterministic:

- A/D/R drag by `100 px` doubles or halves the duration. Pointer motion is accumulated
  piecewise using the modifier state of each move event. Shift scales only subsequent
  motion by `0.1`; pressing or releasing Shift without moving cannot change the preview.
- S drag changes the press-origin level by `0.1 dB/px`; Shift changes it by
  `0.01 dB/px`, using the same piecewise rule. Sustain remains clamped at `0 dB` and
  otherwise uses model validation.
- Up/Right and Down/Left adjust A/D/R multiplicatively by `1%`, or `0.1%` with Shift.
  They adjust S by `0.1 dB`, or `0.01 dB` with Shift.
- Duration proposals are clamped to the minimum renderable duration of one frame under
  the current `RenderConfig`. Text entry remains available for large jumps and exact
  values.

Each accepted stage-value arrow event attempts one complete-patch commit immediately;
accepted auto-repeat events behave the same way. A clamped or unchanged result is a no-op
and emits no commit. Curve-handle arrow steps and Home likewise attempt one immediate
complete-patch commit per accepted key event. Pointer interactions remain preview-during-
drag and single-commit-on-release because they are continuous gestures.

Double-click, Enter, or F2 opens one transient editor directly over the focused stage
label. Time input accepts an explicit `ms` or `s` suffix; a bare value uses the unit shown
by that label. Sustain accepts `dB`. Enter attempts a complete-patch commit and Escape
restores the latest valid authored value. Canonical display uses milliseconds below one
second and seconds at or above one second. The transient editor retains the same strict
parser and exact-value cache as the former numeric control: programmatic updates do not
emit, valid typed values survive display rounding, unchanged commits retain their exact
value, and invalid commits preserve the entered text with one field-specific error.

The graph labels are the only persistent ADSR values. Sustain always shows its actual
level, such as `S -6 dB`; it is never labeled `hold`. Sustain is the constant level held
after Decay, not a separate Hold stage.

Each inline label is a real focusable child control rather than only painted text or an
undocumented hit region. Hover gives it a vertical-adjust cursor and concise tooltip;
focus and active-drag treatments remain visible against the graph. A press focuses and
arms the control. Scrubbing begins only after the platform drag threshold; release below
that threshold is a no-op. An accepted double-click cancels the armed gesture without a
preview or commit and opens exact editing. Once dragging, both stage controls and curve
handles retain pointer capture outside their bounds. Release commits once; Escape,
capture loss, or window deactivation restores the press-origin value and emits no patch
commit. Window deactivation cancels the unfinished interaction before the existing
force-stop path runs.

The transient editor is the only text-input state. Space belongs to text entry while it
is open and cannot trigger Play. Enter commits, Escape cancels, and losing focus cancels
without an implicit patch change. A rejected Enter commit keeps editor focus and retains
its exact text and field-specific error so the user can correct it or press Escape. Enter
or Escape returns focus to the originating inline control; focus-loss cancellation does
not reclaim focus from the intended destination. Opening another transient editor first
cancels the old one and then focuses the new editor.

Graph handles are vertically constrained. Their horizontal Bézier control coordinate is
always the stage midpoint and cannot be dragged. Moving a handle updates a local draft;
releasing it attempts one complete-patch commit. Double-clicking a handle resets only that
stage to linear. Thin guide segments from `P0` to `P1` and from `P1` to `P2` make clear
that the draggable control point is a Bézier handle and need not lie on the rendered
curve. Its precise value appears contextually rather than in a permanent curvature text
box. Handle position is a projection of the model value, using
`c = 2 * (P1.y - (a + b) / 2) / (b - a)` whenever the stage has a nonzero amplitude
span; the graph never stores a second copy of curvature. There is no separate all-curves
reset; the contextual Reset action below owns the complete-envelope reset.

The authoring preview depicts a complete nominal envelope: Attack from zero to full
scale, Decay to the configured sustain amplitude, a fixed-width Sustain shelf, and
Release from that sustain amplitude to zero. It does not claim to visualize an early
note-off from an in-progress Attack or Decay. At an amplitude span too small to resolve
visually, keyboard adjustment and the contextual numeric readout remain available.

The graph's y-axis is linear full-scale amplitude from zero to one. Because a 1 ms Attack
would be unusably narrow beside a 600 ms Decay on a linear time axis, the authoring graph
uses an explicit display transform:

- Sustain receives exactly `16%` of the normalized graph width.
- Attack, Decay, and Release each receive a base `14%` of graph width.
- The remaining `42%` is distributed among Attack, Decay, and Release in proportion to
  `ln(1 + stage_frames)` under the current `RenderConfig`.
- Stage boundaries and inline value labels make the transform clear.
- The display x-coordinate is never used as the DSP's physical time input.

Stage boundaries are not duration controls. Dragging them horizontally would imply a
one-to-one relationship between painted x-position and physical time that this coupled
logarithmic authoring projection deliberately does not have.

The contextual `Reset` action attempts one atomic candidate whose envelope is exactly
`EnvelopeConfig()`:

- Attack `1 ms`;
- Decay `600 ms`;
- Sustain `-6 dB`;
- Release `600 ms`;
- Attack, Decay, and Release curvature `0.0`.

Reset preserves the selected frequency, oscillator, output gain, audio-device state, and
every future patch field outside `EnvelopeConfig`. Like any accepted patch edit, it
clears capture when the replacement is admitted to the command path. Normal
active/releasing deferred-apply rules still govern when the reset envelope is first
auditioned.

Reset closes and discards any incomplete or invalid transient edit, clears only the
envelope-editor error category, and proposes the default envelope once. When the latest
valid authored envelope already equals the defaults and no invalid draft exists, Reset is
a no-op and emits no patch command.

This is an authoring projection, not an oscilloscope. The measured Waveform remains the
physical-time view.

## Keyboard and accessibility behavior

- The four inline stage controls are focusable in A/D/S/R order and expose stage name,
  exact value, unit, and concise drag/type instructions.
- Arrow keys adjust the focused stage; Shift uses the fine step. Enter or F2 opens exact
  editing and Escape cancels the current interaction.
- Each curve handle is reachable by keyboard as an adjustable value.
- Left/Down and Right/Up change curvature by `0.01`; Shift changes it by `0.001`.
- Home resets the focused curve to `0.0`.
- Escape reverts the current draft interaction.
- The graph exposes stage name, current curvature, minimum, maximum, and step through Qt
  accessibility interfaces.
- Every stage value and curve handle is a separate focusable accessible child with an
  adjustable role, exact current value and unit, applicable bounds and step, instructions,
  and value-change events. The transient editor is named for its stage and unit. Every
  child has at least a `24 x 24` device-independent hit and focus target.
- Graph focus order is A value, A curve, D value, D curve, S value, R value, R curve.
- Reset, Load, and Save As retain non-Space keyboard routes. Reset belongs to the
  Envelope context and does not imply pitch reset.
- Global Space remains press-and-hold Play outside text editors and cannot strand a gate
  when focus changes.

## Workbench layout

The top transport remains compact. The main content row becomes:

```text
┌──────────────────────── frequency transport ─────────────────────────┐
├──────── Waveform ────────┬──────── Spectrum ────────┬── Envelope ───┤
│                          │                           │                 │
│ physical-time capture    │ log-frequency capture     │ graph controls  │
│                          │                           │ patch actions   │
└──────────────────────────┴───────────────────────────┴─────────────────┘
```

The Envelope inspector has a fixed width of `288 px`; the two measurement plots
share the remaining width in their existing `4:6` ratio. It spans the plot height rather
than becoming a tall bottom drawer. At `1024 x 640`, both pyqtgraph ViewBoxes retain at
least `320 px` of data-canvas height, all inspector controls remain contained, and no
horizontal scrollbar is introduced. The default remains `1280 x 720`.

The existing bottom facts row is removed. Its editable values and remaining oscillator/
output facts move into the inspector. The measurement header, Clear action, and concise
error banner remain above the main row.

## Frequency-knob redesign

The knob's domain behavior is frozen:

- runtime-tuned C2-C4 range with C3 at the logarithmic midpoint;
- continuous unsnapped hertz values;
- relative vertical drag with no click jump;
- dynamic Shift fine mode during drag;
- one-cent wheel and arrow steps, `0.1` cent with Shift+arrow;
- double-click center reset;
- clamping without wrap;
- live and release retuning without phase or envelope retrigger;
- direct-Hz entry synchronization without feedback loops;
- `QAccessible.Dial` value and actions.

Only presentation and discoverability change:

- The dial remains an `88 x 88 px` compact transport control.
- The hollow progress-ring appearance becomes a solid, restrained pro-audio dial face.
- A pointer communicates rotation more clearly than a floating status dot.
- The sweep runs conventionally from lower-left through top to lower-right, so C3 is at
  twelve o'clock.
- C2, C3, and C4 landmarks are visible; the center notch is emphasized.
- Hover and pressed/dragging states are distinct without animation noise.
- Hover uses a vertical-resize/drag cursor.
- Keyboard focus is integrated into the outer ring rather than drawn as a dominant
  square tile.
- A visible `Frequency` group ties the dial, exact-Hz field, and derived note/cents label
  together.
- A concise tooltip explains vertical drag, Shift fine mode, wheel/arrows, and
  double-click reset.

No third-party knob package, bitmap skin, SVG asset, or plugin framework is required.
The custom Qt widget remains small and purpose-built.

## Error presentation and atomicity

- Syntactically incomplete transient editor text remains local and does not flash global
  errors on every keystroke.
- A failed commit identifies the exact inline stage beside the editor and uses the
  existing single error surface for a concise full explanation.
- `Pending` has display precedence over `Editing`; otherwise any local draft or field
  error displays `Editing`, and only a clean applied patch displays `Active`. Pending does
  not clear or hide a concurrent invalid transient edit.
- A rejected commit exposes its message through the transient editor's accessible
  description and emits one accessibility announcement. Status transitions emit the
  corresponding accessible state-change event.
- A validation failure leaves authored patch, pending/applied state, capture generation,
  gate, and audio command stream unchanged.
- A valid commit clears only editor-originated validation errors.
- File and device errors remain distinguishable and are not hidden by a later editor
  refresh.
- No error path creates a partial v2 patch.
- Repeated identical device failures remain deduplicated.

## Baseline hardening carried into Milestone B

Before changing the envelope model, Milestone B closes the parked analysis-validation
edge from the Milestone A review:

- `AnalysisConfig(waveform_window_seconds=1e308)` must be rejected by composed analysis
  validation with a named `ValueError`.
- Duration-to-frame multiplication is checked for finiteness before `floor`.
- The fix is isolated and does not change ordinary analysis results.

This is a prerequisite cleanup task, not a new Milestone B feature.

## Testing strategy

### Pure curve and model tests

- Endpoint values are exact for every stage and curve extreme.
- `c = 0` matches all Milestone A linear characterization vectors sample-for-sample.
- Curves are monotone and bounded across a dense parameter/time sweep.
- Sign semantics are consistent for rising and falling segments.
- Invalid type, boolean, non-finite, and out-of-range cases name the correct field.
- Preview and float64 renderer-envelope sampling agree at shared normalized positions
  with `rtol = 0` and `atol = 1e-12`.

### Envelope renderer tests

- Attack, Decay, Sustain, early Release, repeated note-off, and retrigger behavior remain
  deterministic.
- Release begins from the exact last-emitted level for every source stage and curvature.
- Rendering the same event sequence under different block partitions produces the same
  float32 output bit-for-bit; the zero-curvature path also retains exact legacy output.
- Segment frame rounding and one-frame segments have pinned outputs and transitions.

### Codec tests

- Strict v1 remains readable and maps to three zero curvature values.
- Canonical v2 output round-trips exactly.
- V1 loaded then saved becomes canonical v2.
- Per-version key sets are closed and cannot be mixed.
- Duplicate keys, trailing content, booleans, huge integers, non-finite values, and curve
  bounds are rejected with field context.
- Atomic overwrite failure preserves the previous destination for v2 files.

### Controller and audio-boundary tests

- Idle authoring sends one replacement and clears capture.
- Held/releasing authoring sends no immediate replacement.
- Multiple valid edits coalesce to the latest patch and send one replacement on matching
  idle.
- Renderer-idle does not notify while final Release PCM remains in replaceable staging;
  deferred replacement cannot truncate that tail.
- Stale idle callbacks cannot apply a pending patch.
- Play cannot retrigger an old patch while replacement is pending.
- File Load remains immediate and discards a pending editor patch atomically.
- Device failure, recovery, and shutdown preserve the latest valid authored patch without
  duplicate reset/replacement commands.
- Immediate/force-stop patch replacement still preempts staged old PCM; a deferred
  editor replacement reaches the command path only after its release watermark drains.

### Native UI tests

- Inline labels show canonical A/D/R units and the actual Sustain dB value; `S hold` is
  impossible.
- Transient editors preserve exact values across display rounding and programmatic
  updates, commit with Enter, and restore with Escape.
- Pointer scrubbing and exact editing create identical immutable patches.
- Scrubbing emits drafts but only one complete commit on release, remains anchored to the
  press value during graph relayout, and does not treat x-position as physical time.
- Scrubbing retains pointer ownership outside the label; Escape and deactivation revert
  without a commit. Shift changes sensitivity during an existing gesture.
- Inline controls have real focus/accessibility semantics and deterministic Tab order;
  Enter/F2 editing owns Space, focus-loss cancellation is silent, explicit Enter/Escape
  returns focus to the originating control, and focus loss preserves its intended target.
- Real pointer-event tests distinguish click, double-click, and threshold-crossing drag;
  cover capture loss/out-of-bounds release for stage controls and curve handles; and prove
  that zero-motion Shift transitions cannot jump a preview.
- Accessibility tests assert each child's role, name, value, unit, hit target, order, and
  value-change events, plus the transient editor's error announcement and focus behavior.
- Handle bounds, reset behavior, keyboard stepping, focus, and accessibility are real
  widget-event tests.
- Curve values appear contextually and no permanent curvature text box remains.
- Reset restores the exact default `EnvelopeConfig` in one candidate while preserving
  frequency and non-envelope patch fields; capture follows the ordinary admitted-edit
  invalidation contract.
- Loading v1/v2 refreshes all inline labels and handles without feedback loops.
- Active, Editing, Pending, Pending-plus-invalid, file-error, and device-error states
  follow the declared status/error precedence.
- Actual plot ViewBoxes remain at least `320 px` high in healthy and error states at the
  minimum window size.
- Knob tests retain every domain interaction and add orientation, pointer, landmarks,
  hover, drag cursor, focus, tooltip, and accessibility probes.

## Native acceptance

Milestone B acceptance requires:

1. The full automated suite, Ruff lint, Ruff format check, silent public-import smoke,
   and base-to-head diff check pass.
2. A v1 default patch loads, displays three linear handles, and saves as canonical v2.
3. Curved Attack, Decay, and Release are audibly distinguishable when auditioned on the
   next Play without changing the prior held/releasing sound.
4. Multiple edits during one audition produce only the final pending patch.
5. Load, invalid edit, device failure/recovery, and close leave no stuck sound or partial
   editor state.
6. The knob feels discoverable and precise while preserving continuous live/release
   tuning.
7. Complete native idle, editing, pending, and live screenshots show the outer frame and
   contain no private desktop data.
8. The full workbench is visually inspected at nominal `125%` and `100%` Windows scaling,
   at default and minimum client sizes.
9. Production review finds no Qt dependency in core synthesis/codec mathematics, no
   duplicate preview formula, no v1/v2 compatibility alias in the runtime model, and no
   dead Milestone A patch-facts surface.
10. The Envelope inspector has no permanent ADSR or curve-entry grid: A/D/S/R are
    adjusted and edited on the graph, Sustain reports its actual dB level, and Reset
    restores only the complete envelope defaults.

## Milestone boundary after completion

After Milestone B, Harpy has a deterministic sine patch authoring surface and a native
measurement workbench. That configuration API may generate controlled patch variants for
later experiments, but Milestone C still begins with a fresh design for the sine-only
retuning Gym.

Milestone C must independently decide:

- whether the actor receives audio, numerical observations, tools, or a combination;
- target specification and leakage boundaries;
- octave, semitone, and cents action granularity;
- episode termination and success tolerances;
- reward construction and quality penalties;
- training versus benchmark case generation;
- local-policy and external-model adapter contracts;
- reproducibility, compute reporting, and evaluation artifacts.

None of those decisions are smuggled into the Milestone B editor.
