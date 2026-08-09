# Milestone B acceptance evidence

Date exercised: 2026-08-08

Implementation range: `7d6fd5e9b951e391c683196f84f2b0de2a46cdaa...2c80da7edd92a2b0c275ac00f715729d6a1b8edf`
(`origin/main...2c80da7`)

Environment: WSLg, Python 3.12.3, PySide6/Qt 6.11.1,
`PULSE_SERVER=unix:/mnt/wslg/PulseServer`, Windows `AppliedDPI=120`

## Acceptance status

The fresh automated gates, structural contracts, exact native WSLg launch, Pulse route,
native accessibility tree, and a bounded momentary Play/PCM observation passed. These
results accept the graph-native implementation on those measured dimensions.

Full manual acceptance remains incomplete. Subjective audibility, a sustained native
hold/edit/release interaction, a real physical audio-device loss, complete outer-frame
visual inspection at nominal 125% and 100% Windows scaling, and the four qualifying
screenshots remain pending. No claim below substitutes automated proof for a native
observation or a Pulse measurement for human hearing.

## Automated evidence

The clean gates ran from the feature worktree between
`2026-08-08T22:54:18-05:00` and `2026-08-08T22:54:44-05:00`.

| Command | Exit | Exact observed result |
| --- | ---: | --- |
| `uv run pytest` | 0 | `11054 passed in 23.83s`; the suite also emitted expected Qt environment diagnostics for unavailable PipeWire symbols and test mouse grabbing. |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run ruff format --check .` | 0 | `63 files already formatted` |
| `uv run python -c "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; import harpy.gui.app; import harpy.gui.envelope_stage_control; import harpy.synth.curves, harpy.synth.engine, harpy.synth.patch_json"` | 0 | Zero stdout and zero stderr. |
| `git diff --check origin/main...HEAD` | 0 | Zero output. |
| `find src/harpy -type f -name '*.py' -print0 \| xargs -0 wc -l` | 0 | 5,454 total production Python lines; per-module results are recorded below. |

The 11,054 tests are automated evidence for deterministic curve/DSP behavior, exact
endpoints, v1/v2 codec strictness and atomic saves, immutable patch validation,
generation and PCM-watermark handling, deferred/coalesced patch admission, capture
retention, load/reset/clear/device/shutdown lifecycles, graph projection, scrub and
transient-entry behavior, focus/Space routing, accessibility contracts, native layout
constraints, and frequency-dial interactions. They are not presented as direct native
or human observations.

## Structural evidence

Each required structural contract returned exit 0 with zero output:

```bash
! rg -n 'attackEntry|decayEntry|sustainEntry|releaseEntry|curveEntry|resetCurvesButton|resetCurvesAction|Reset curves to linear|S  hold|EnvelopeFieldKind\.CURVATURE' src/harpy tests
! rg -n 'LinearEnvelope|patchFacts|_patch_value_labels|_set_patch_facts' src/harpy
! rg -n 'Gymnasium|gymnasium|reward|actor adapter|model adapter' src/harpy
! rg -n 'from PySide6|import PySide6|pyqtgraph' src/harpy/synth src/harpy/analysis.py src/harpy/capture.py src/harpy/playback.py src/harpy/tuning.py
```

Production-module inspection recorded these line counts and responsibilities:

| Module | LOC | Principal responsibility |
| --- | ---: | --- |
| `analysis.py` | 230 | Pure waveform/spectrum analysis and validation |
| `capture.py` | 245 | Generation-safe sample history and retained observations |
| `config.py` | 20 | Validated application composition defaults |
| `playback.py` | 76 | Typed controller-to-audio commands |
| `tuning.py` | 77 | Hertz/MIDI conversion and note/cents descriptions |
| `gui/app.py` | 89 | Runtime composition and native entry point |
| `gui/envelope_editor.py` | 458 | Sole authored patch/draft owner, validation, status, Reset, and file intents |
| `gui/envelope_entry.py` | 203 | Transient exact duration/decibel parsing and error lifecycle |
| `gui/envelope_graph.py` | 784 | Graph composition, stage/handle projection, geometry, and field proposals |
| `gui/envelope_stage_control.py` | 522 | Stage scrubbing, keyboard/exact interaction, and accessibility bridge |
| `gui/frequency_entry.py` | 110 | Exact frequency entry |
| `gui/frequency_knob.py` | 360 | Continuous logarithmic dial interaction, painting, and accessibility |
| `gui/patch_dialogs.py` | 30 | Native patch-file dialog port |
| `gui/qt_audio.py` | 434 | Cohesive Qt platform adapter for PCM, device, recovery, and sink lifecycle |
| `gui/signal_views.py` | 147 | Scientific waveform and spectrum views |
| `gui/window.py` | 502 | Workbench layout, user actions, and view/controller binding |
| `gui/workbench_controller.py` | 295 | Transport, patch admission, capture, and audio availability state |
| `gui/workbench_spec.py` | 20 | Native frequency bounds |
| `synth/curves.py` | 185 | The single quadratic curve formula and envelope preview sampler |
| `synth/engine.py` | 116 | Deterministic monophonic renderer and performance state |
| `synth/envelope.py` | 140 | ADSR state machine using the shared curve evaluator |
| `synth/models.py` | 158 | Immutable oscillator/envelope/patch/render models and validation |
| `synth/patch_json.py` | 243 | Strict v1/v2 parsing and canonical atomic v2 writing |
| package `__init__.py` files | 10 | Package markers and synth re-exports |

The source has one patch truth: `EnvelopeEditor` owns the authored patch and local
immutable envelope draft. `EnvelopeGraph` owns composition and proposal-only
projection. `EnvelopeStageControl` owns interaction/accessibility and reads its value
through an owner callback. Both graph preview and DSP import the formula from
`synth.curves`; inspection found no second curve implementation, duplicate patch truth,
dead compatibility branch, or premature Gym code. `qt_audio.py` remains one cohesive
platform adapter and was not split solely because it is large.

## Native WSLg observations

The exact application command was launched from the tested worktree with the inherited
Pulse route preserved:

```bash
XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir uv run harpy
```

Pulse reported exactly one sink input with `application.name == "harpy"`. The required
PID/cwd/cmdline proof exited 0. For the first observed launch it resolved PID `966839`
to this worktree and printed:

```text
/home/haydenw/Projects/Harpy/.worktrees/native-sine-lab/.venv/bin/python3 /home/haydenw/Projects/Harpy/.worktrees/native-sine-lab/.venv/bin/harpy
```

The stream was `float32le 2ch 48000Hz` with channel map
`front-left,front-right`. A second exact launch used PID `968972` and sink-input index
25 and passed the same worktree and format assertions. Only the exact proven PID was
sent `SIGTERM`; `/proc/968972` disappeared and a fresh Pulse query asserted that no
`harpy` input remained. No broad process kill was used.

Windows exposed the native window as `Harpy (Ubuntu)`. The Linux AT-SPI bridge directly
reported a Qt 6.11.1 frame with client extents 1280 x 720. Its graph subtree exposed
four adjustable spin-box roles and three slider roles in the native process:

| Accessible control | Role | Description / target |
| --- | --- | --- |
| Attack duration | spin box | milliseconds or seconds; vertical drag; Enter/F2 exact edit; 50 x 24 px observed target |
| Decay duration | spin box | same duration contract; 75 x 24 px target |
| Sustain level | spin box | decibels; vertical drag; Enter/F2 exact edit; 38 x 24 px target |
| Release duration | spin box | duration contract; 76 x 24 px target |
| Attack, Decay, Release curve | slider | each 24 x 24 px target |
| Frequency | dial | native Dial role with drag/fine/wheel/arrow/reset description; 88 x 88 px target |
| Reset | push button | `Reset envelope to defaults (Ctrl+R)`; 266 x 36 px target |

The idle curve-readout label existed with empty text and zero extents, so it was not a
permanent visible field. AT-SPI did not expose the painted A/D/S/R display strings as
text properties, so the native run does not independently prove their exact idle copy;
the `A 1 ms`, `D 600 ms`, `S -6 dB`, `R 600 ms` and no-`hold` contracts are automated
evidence from the graph/widget tests.

A bounded AT-SPI `Press` action on Play directly observed the native measurement state
change from empty to `Measuring…`, then `Live`, then `Captured`, while patch status stayed
`Active`. A monitor attached to that verified sink input captured 95,210 float32 stereo
frames: every sample was finite, 65,774 samples were nonzero, and the absolute peak was
`0.250854492`. This proves native routed signal production; it does not prove audibility.

Attempts to synthesize a sustained native hold with XTest and the AT-SPI device-event
controller did not open the Play gate; the bounded monitor returned only zeros. Those
attempts are recorded as harness limitations, not product passes or failures. Therefore
native held-note editing, deferred Pending status, coalescing, release retention, next-
Play admission, transient exact entry, graph scrubbing, curve contextual behavior,
load/save/reset/clear, deactivation, simulated device recovery, and close interaction
remain supported by automated evidence but were not independently re-observed in this
native session. A real physical device loss was not induced.

## Screenshot evidence

No screenshot assets were created. The Computer Use runtime failed before control or
capture with this exact error:

```text
codex/sandbox-state-meta: sandboxCwd is not a local file URI: file:///home/haydenw/Projects/Harpy
```

Because complete native outer-frame capture could not be qualified, no client-only or
placeholder image was committed for idle, editing, pending, or live state. No private
desktop data was captured.

## Pending manual and visual checks

- **Human hearing:** a person must confirm subjective audibility, continuous retuning,
  unchanged held/releasing audio during edits, the complete release tail, and absence
  of stuck sound after lifecycle actions.
- **Sustained native graph interaction:** directly exercise and observe vertical stage
  scrubbing, dynamic Shift, arrow auto-repeat, transient exact editing and focus rules,
  curve contextual readout, v1 Load/v2 Save As, Pending coalescing, next-Play admission,
  envelope-only Reset, atomic invalid edits, capture retention, Clear, simulated device
  recovery, deactivation, and close. The automated suite is green, but the native input
  harness could not sustain the gate.
- **Physical device loss:** unplug or otherwise remove the real selected output and
  observe recovery. Only automated/simulated recovery is presently proven.
- **Nominal 125% complete-frame fit:** Windows reported `AppliedDPI=120`, but a
  DPI-aware complete outer-frame inspection at 1280 x 720 and 1024 x 640 could not be
  captured.
- **100% scaling:** not exercised. Changing the user's Windows display scaling was
  outside granted authority.
- **Qualifying screenshots:** capture complete native outer frames with no private
  desktop data for idle, transient editing, retained old capture plus Pending intent,
  and live states at the four declared asset paths.

Milestone C remains stopped at its Gym brainstorming/specification boundary. No
Gymnasium environment, action/observation/reward schema, leakage control, adapter, or
benchmark implementation was started during this acceptance work.
