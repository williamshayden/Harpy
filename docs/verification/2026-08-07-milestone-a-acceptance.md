# Milestone A acceptance evidence

Date exercised: 2026-08-08

Implementation base: `5cdcd13` (`fix: harden native workbench interaction boundaries`),
plus the recorded Task 10 review fixes

Environment: WSLg, 48 kHz `RDPSink`, Python 3.12, PySide6/Qt 6.11.1

## Acceptance status

The automated, structural, and safely automatable native checks below have fresh
evidence. Milestone A is **not declared complete** because subjective hearing, a
complete-outer-frame visual check at the current nominal 125% Windows scale and at
100%, and qualifying idle/live native screenshots remain pending.

This document distinguishes application observations from evidence produced by unit
tests. A pass is recorded only where the named behavior was observed in the cited run.

## Automated evidence

All commands ran from the Milestone A worktree on 2026-08-08.

| Command | Observed result |
| --- | --- |
| `uv run pytest` | Exit 0; 9,353 tests passed. Qt reported that PipeWire was unavailable where the full suite deliberately constructed the real audio backend. |
| `uv run ruff check .` | Exit 0; `All checks passed!` |
| `uv run ruff format --check .` | Exit 0; 50 files already formatted. |
| `uv run python -c "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; import harpy.gui.app; import harpy.synth.engine, harpy.synth.patch_json"` | Exit 0 with 0 stdout bytes and 0 stderr bytes. |
| `git diff --check origin/william/native-sine-lab...HEAD` | Exit 0 with no findings before the documentation edit. |

The tests cover engine and envelope determinism, strict patch JSON, tuning, calibrated
analysis, generation-safe capture, semantic controller behavior, Qt audio format and
device lifecycle boundaries, workbench interactions, and layout contracts. They are
supporting automated evidence; the native observations below were gathered separately.

Independent review found that the first acceptance commit's exact import smoke exited 0
but emitted 135 stderr bytes because `harpy.gui.app` eagerly imported Qt Multimedia.
A subprocess regression for the exact command was added first and observed failing on
the stderr assertion. `harpy.gui.app` now resolves `QtAudioBackend` only when default
runtime construction needs it; explicit injected factories remain eager-free. The
regression and exact command then passed with both streams empty.

## Structural evidence

`find src/harpy -type f -name '*.py' -print0 | xargs -0 wc -l` reported 2,914 production
Python lines. Inspection found one principal responsibility per module and no legacy
browser, GUI-bound DSP model, MIDI-only controller, or compatibility path in the
production source.

| Module | LOC | Principal responsibility |
| --- | ---: | --- |
| `tuning.py` | 77 | Hertz/MIDI-coordinate conversion and note/cents descriptions |
| `playback.py` | 76 | Typed commands between controller and audio adapter |
| `capture.py` | 245 | Bounded generation-safe sample history and capture state |
| `analysis.py` | 226 | Pure waveform/spectrum observation and peak measurement |
| `gui/app.py` | 88 | Runtime composition, lazy default audio resolution, and application entry point |
| `gui/patch_dialogs.py` | 30 | Native patch-file dialog port and adapter |
| `gui/window.py` | 456 | Workbench widgets, layout, formatting, and user actions |
| `gui/workbench_controller.py` | 247 | Semantic transport, frequency, patch, and capture state |
| `gui/workbench_spec.py` | 20 | GUI frequency bounds derived from tuning |
| `gui/frequency_entry.py` | 110 | Exact-hertz entry validation and commit behavior |
| `gui/signal_views.py` | 145 | Fixed scientific waveform and spectrum presentation |
| `gui/qt_audio.py` | 418 | Qt device negotiation, pull source, PCM, and sink lifecycle |
| `gui/frequency_knob.py` | 249 | Logarithmic accessible continuous-frequency control |
| `config.py` | 20 | Validated application composition defaults |
| `synth/patch_json.py` | 182 | Strict versioned patch serialization |
| `synth/engine.py` | 91 | Deterministic monophonic oscillator and performance state |
| `synth/envelope.py` | 106 | Linear-amplitude ADSR state and rendering |
| `synth/models.py` | 118 | Immutable patch/render models and validation |
| package `__init__.py` files | 10 | Package markers and the small synth re-export surface |

The two largest files remain cohesive platform or presentation boundaries rather than
mixed research-domain modules: `window.py` owns the native view, and `qt_audio.py` owns
the Qt audio adapter and device lifecycle.

## Native WSLg observations

### Launch and audio route

No existing Harpy process was present. The application was launched from the tested
worktree with the required runtime override while preserving
`PULSE_SERVER=unix:/mnt/wslg/PulseServer`:

```bash
XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir uv run harpy
```

The exact entry point created one Pulse sink input owned by the verified Harpy process:
`float32le`, two device channels, 48,000 Hz, `application.name = "harpy"`, uncorked,
against `RDPSink`. The process and child executable both resolved to the tested
worktree. After this launch check, only those exact verified PIDs were stopped; no broad
process kill was used.

The remaining checks used the same `build_runtime` composition, real
`QtAudioBackend`, visible WSLg window, and real Pulse route. Qt input events exercised
the widgets. Sustained analysis holds called the window's Play handler after the host
had settled focus so that Codex returning focus to its own window could not trigger
Harpy's intentional `WindowDeactivate` force-stop during evidence collection.
`RDPSink.monitor` captures were bounded, ephemeral, and made only while Harpy was the
sole sink input.

| Native check | Observation | Result |
| --- | --- | --- |
| Default and minimum client geometry | Visible native client reported 1280×720 default and 1024×640 minimum on a 1920×1080 Qt screen. Complete Windows outer-frame fit was not observable. | Partial; visual frame pending |
| Initial selection and Play | C3 was 261.625565301 Hz with label `C3 +0.0¢`. Qt mouse press/release set and cleared the held gate. | Pass |
| Native stream and initial peak | Harpy was the sole 48 kHz float Pulse sink input. After the held generation contained 16,384 rendered frames, the internal peak was 261.671562839 Hz (0.0460 Hz error). The real sink-monitor capture measured 261.671076909 Hz and 0.198120 FS (0.0455 Hz error). | Pass |
| Space | With focus outside the frequency editor, Qt Space press held the same gate and release cleared it. | Pass |
| Continuous/logarithmic retune | A 20-pixel upward drag moved 261.625565301 Hz to the calculated 300.528856486 Hz without changing capture generation. | Pass |
| Stable retuned peak | After 550 ms, more than 16,384 newly rendered 48 kHz frames, the measured peak was 300.498088212 Hz (0.0308 Hz error). | Pass |
| Fine and musical steps | Double-click restored C3; arrows and wheel applied 1-cent steps; Shift+arrow applied 0.1 cent; Shift+10-pixel drag applied 12 cents. | Pass |
| Release retune and silence | An arrow retune during release preserved generation and did not retrigger. The workbench reached Captured with no active voice; the final 4,800 sink-monitor frames had peak 0.000000000 FS. | Pass |
| Capture and Clear | Release retained both plotted observations. Clear returned the capture to Empty and removed both plot traces. Initial/cleared views contained no fabricated data. | Pass |
| Scientific axes | Waveform ranges were 0–50 ms and -1–1 FS. Spectrum ranges were log 20 Hz–20 kHz and -120–0 dBFS. | Pass |
| Patch save/load | Save As emitted canonical version-1 JSON; loading the file round-tripped the patch, preserved selected frequency, stopped performance state, and emptied capture. | Pass |
| Invalid patch atomicity | A string in `envelope.attack_seconds` produced `envelope.attack_seconds must be a JSON number`; patch, selection, capture generation/state, gate, and voice state were unchanged. | Pass |
| Healthy/error copy | Healthy label copy contained no device, channel, sample-format, ready, or healthy status. | Pass |
| Failure/recovery | A safe window/controller simulation delivered the same audio failure twice: one banner remained and Play was disabled. Recovery cleared the error, remained silent/Empty, re-enabled Play, and added no success copy. This did not disconnect the user's real device. | Pass (simulated) |
| Close while held | Closing the native window while its gate was held invoked shutdown; the process's Pulse sink input was absent 500 ms later. | Pass |

These observations prove that the application produced a real routed sine stream and
then real zero output after release. They do not prove that a human heard or judged the
sound.

## Screenshot evidence

No screenshot is attached. The available offscreen/client-only captures do not include
the complete native Windows outer frame and therefore do not qualify. The Windows
Graphics Capture automation runtime failed initialization with a workspace-URI error,
so it could not safely capture a qualifying idle or live frame. No private desktop data
was captured or committed.

## Pending manual and visual checks

- **Human hearing:** confirm that held C3 is subjectively audible and stable, that live
  and release retuning sound continuous without retrigger, that the configured release
  is heard, and that release/close leaves no stuck sound. Pulse stream and monitor
  evidence above is not a substitute for hearing.
- **Current nominal 125% visual fit:** Windows reported `AppliedDPI=120` (nominal 125%)
  and the native client reported its intended geometry, but complete outer-frame fit
  and minimum-size usability were not visually captured. This remains pending.
- **100% visual fit:** not exercised. Changing the user's display scaling was outside
  the granted authority. This remains pending rather than inferred from client size.
- **Qualifying artifacts:** capture one idle and one live default-size screenshot that
  show the complete native outer frame and no private desktop data, then store them in
  `docs/verification/assets/` and reference them here.
Until these four pending checks are resolved, this evidence does not authorize marking
Milestone A complete or making its draft pull request ready.
