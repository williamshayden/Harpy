# Harpy

Harpy is an exploratory, model-agnostic benchmark for agents that manipulate audio with constrained pitch tools until it reaches a symbolic musical goal.

The project is currently in research and design. The living notebook records project-wide hypotheses, references, decisions, and open questions:

- [Project notebook](docs/project-notebook.md)

The native sine lab design is approved and has an implementation plan:

- [Native sine lab design](docs/superpowers/specs/2026-08-07-native-sine-lab-design.md)
- [Native sine lab implementation plan](docs/superpowers/plans/2026-08-07-native-sine-lab.md)

## Native sine lab

The first implementation slice is a native PySide6 application: one deterministic NumPy sine voice, a MIDI-note selector centered on C3/MIDI 60, the fixed 1 ms / 600 ms / -6 dB / 600 ms linear envelope, and waveform/spectrum plots.

```bash
uv sync --dev
uv run harpy
```

Run the automated checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

This slice intentionally contains no browser UI, MIDI, imported-audio editing, RL environment, database, or third-party synth engine.
