"""A waveform adapter using the same public spectrum decoder and committed control.

Run with Python 3.12: python examples/custom_actor.py --output runs/waveform-example.json
This demonstrates an extension point, not a new pitch estimation method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harpy.envs.models import ObservationMode
from harpy.envs.spectrum import encode_log_spectrum
from harpy.experiments import ActorSpec, evaluate, save_result, spectrum_peak_actor_spec, summarize
from harpy.experiments.protocols import smoke_episodes


class WaveformPeak:
    def __init__(self) -> None:
        self._controller = spectrum_peak_actor_spec().factory()
        self._spectrum = None

    def decide(self, observation):
        # Committed control only needs one estimate. Keep the waveform track's
        # initial encoding local to this episode rather than recomputing each step.
        if self._spectrum is None:
            self._spectrum = encode_log_spectrum(observation["waveform"])
        spectrum_view = {key: value for key, value in observation.items() if key != "waveform"}
        spectrum_view["spectrum"] = self._spectrum
        return self._controller.decide(spectrum_view)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = spectrum_peak_actor_spec()
    waveform = ActorSpec(
        name="waveform-spectrum-peak",
        factory=WaveformPeak,
        observation_mode=ObservationMode.WAVEFORM,
        estimator="waveform-to-harpy-spectrum-peak-v1",
        decoder=baseline.decoder,
        controller=baseline.controller,
        artifact_paths=(Path(__file__).resolve(),),
    )
    result = evaluate(smoke_episodes(), (baseline, waveform))
    save_result(result, args.output)
    print(summarize(result), end="")


if __name__ == "__main__":
    main()
