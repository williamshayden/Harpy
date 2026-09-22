"""Real subprocess fixture: park in the third actor after two durable commits."""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path


def pause():
    marker = os.environ.get("HARPY_TEST_PAUSE")
    if marker:
        Path(marker).write_text("inside third member", encoding="utf-8")
        while True:
            time.sleep(1)


class SoundActor:
    def decide(self, observation):
        from harpy.tasks.sound_match import Submit

        if observation.episode_id == "e2":
            pause()
        return Submit(0)


class PitchActor:
    created = 0

    def __init__(self):
        self.index = type(self).created
        type(self).created += 1

    def act(self, observation):
        from harpy.envs.models import PitchAction

        if self.index == 2:
            pause()
        return PitchAction.SUBMIT


if __name__ == "__main__":
    task, output, operation = sys.argv[1:]
    if task == "sound":
        from harpy.tasks.sound_match import ActorSpec, EpisodeSpec, default_patch
        from harpy.tasks.sound_match.durable import evaluate_bundle

        episodes = [
            EpisodeSpec(f"e{i}", replace(default_patch(), notes_cents=(6100 + i * 20,)))
            for i in range(4)
        ]
        actors = [ActorSpec("worker", SoundActor)]
        options = {}
    else:
        from harpy.envs.models import ObservationMode
        from harpy.experiments import ActorSpec, EpisodeSpec
        from harpy.experiments.durable import evaluate_bundle

        episodes = [EpisodeSpec(f"e{i}", 6000 + i * 20, 12) for i in range(4)]
        actors = [ActorSpec("worker", PitchActor, ObservationMode.SPECTRUM, "none", "none", "test")]
        options = {"trace": True}
    result = evaluate_bundle(
        episodes, actors, Path(output), resume=operation == "resume", **options
    )
    assert result.complete and result.completed == 4
    assert "torch" not in sys.modules and "stable_baselines3" not in sys.modules
