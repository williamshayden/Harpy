# Harpy Milestone D Learned Sine Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Train, persist, evaluate, and run Harpy's first learned single-sine tuning
policies: a supervised oracle-label diagnostic followed by five fresh PPO seeds on the
unchanged `Harpy/SinePitch-v0` contract.

**Architecture:** A new Qt-free `harpy.learning` package wraps the frozen Gym
environment without copying its transition or reward logic. Deterministic suite
generation, a bit-identical spectrum cache, one normalized two-branch policy network,
terminal-only evaluation, and strict local artifacts are framework-independent where
possible; PyTorch and Stable-Baselines3 enter only through lazy training modules in an
optional dependency group. BC and PPO share architecture and evaluation contracts but
never weights or optimizer state.

**Tech Stack:** Python 3.12, NumPy 2.x, Gymnasium 1.3.x, PyTorch 2.8+, Stable-Baselines3
2.9.0, existing Harpy synth/spectrum/planning core, pytest, Ruff, uv.

## Global Constraints

- The approved design is
  `docs/superpowers/specs/2026-08-10-harpy-milestone-d-learned-sine-policy-design.md`;
  its values, nonclaims, leakage boundaries, and artifact rules govern every task.
- Create the implementation branch from approved commit `957c140` in an isolated
  worktree via `superpowers:using-git-worktrees`; never alter, stage, stash, or discard
  the user's root-checkout edit in `docs/project-notebook.md`.
- Learned training and actor evaluation use only the unchanged seven-action,
  64-step, `gamma=1.0` `Harpy/SinePitch-v0` contract. Do not change action IDs,
  rewards, controls, spaces, reset options, registration kwargs, or terminal results.
- The registered base environment retains direct rendering and `_candidate_audio`.
  Only a nonregistered learning subclass may omit retained audio and cache spectra.
- Actor input contains only the current `(1961,)` float32 spectrum and five normalized
  public scalars. It never contains source/current/target pitch truth, cents error,
  FFT peak, suite identity, oracle actions, or `EpisodeResult`.
- BC may use hidden truth only to create oracle labels and to index evidence solely for
  reconstructing the exact actor-visible spectrum; it never enters scalar features or
  serialized actor input. PPO starts from fresh seeded weights and never imports or
  loads a BC checkpoint.
- PPO must suppress SB3 timeout-value bootstrapping for Harpy
  `BUDGET_EXHAUSTED` while preserving the raw Gym truncation contract.
- The fixed suites are exact: smoke 32 episodes/seed `202608100`, IID 256/seed
  `202608101`, and register-OOD 256/seed `202608102` with 128 lower and 128 upper
  sources. Fixed and BC pairs are unique and disjoint as specified.
- The cache key range is integer cents `1100..10900`; it stores at most 9,801
  immutable float32 spectra and no rendered audio.
- Optional training dependencies are exactly `stable-baselines3==2.9.0` and
  `torch>=2.8,<3` in dependency group `train`. Ordinary Harpy imports remain
  Torch/SB3-free.
- CPU is default and authoritative. CUDA is explicit, recorded, exploratory, and
  criterion-ineligible.
- Checkpoint BC uses seed 0. Headline PPO uses exactly seeds `0..4`; extra seeds are
  exploratory and never substitute for the declared set.
- Every artifact path is explicit, new, create-only, local, schema-v1, incomplete
  first, complete last, hash checked, and closed to unknown files. Do not add a
  database, tracker service, remote upload, or resume protocol.
- Random, Reward Search, Spectrum Peak, and Oracle remain separately labeled controls;
  capability lanes are never pooled.
- Production changes follow strict RED → observe the intended failure → minimal GREEN
  → refactor. Do not weaken existing Milestone C assertions to admit the new code.
- Each task commits only its scoped files after focused tests, the full suite, Ruff
  lint, Ruff format, and `git diff --check` pass. Training-dependent tasks run with
  `uv sync --group train` and prove their marked tests did not skip.

## Execution Topology

Use linked task worktrees/commits so independent work actually runs in parallel:

```text
Wave 1: Task 1
Wave 2: Task 2 | Task 4 | Task 6
Wave 3: Task 3 (after Task 2)
Wave 4: Task 5 (after Tasks 3 and 4)
Wave 5: Task 7 | Task 8 (after Tasks 5 and 6)
Wave 6: Task 9
Wave 7: Task 10
Wave 8: Task 11
```

After Task 1, Tasks 2, 4, and 6 are independent lanes. Task 3 follows Task 2. Task 5
waits for Tasks 3 and 4; Tasks 7 and 8 wait for both Task 5 and Task 6, then run in
parallel from the same verified base. Task 9 is the first cross-trainer integration
task. Every merge boundary reruns the combined focused slice before the next wave.

---

## Final File Map

| File | Responsibility |
| --- | --- |
| `src/harpy/learning/__init__.py` | Lightweight package marker and deliberate dependency-free exports. |
| `src/harpy/learning/models.py` | Immutable profiles, episode/suite records, trainer and evaluation enums. |
| `src/harpy/learning/suites.py` | Fixed suites, BC splits, PPO indexed training sequence, canonical digests. |
| `src/harpy/learning/cache.py` | Bounded mutation-detecting spectrum cache. |
| `src/harpy/learning/envs.py` | Cached env subclass, scheduled-reset wrapper, learning env factories. |
| `src/harpy/learning/errors.py` | Lightweight contract, dependency, artifact, and execution errors. |
| `src/harpy/learning/dependencies.py` | Lazy Torch/SB3 loading and exact install guidance. |
| `src/harpy/learning/observations.py` | Raw Gym validation and two-Box policy preprocessing. |
| `src/harpy/learning/network.py` | Shared 1D-conv/scalar feature trunk and BC/SB3 adapters. |
| `src/harpy/learning/actors.py` | Narrow learned actor protocol and loaded BC/PPO actor adapters. |
| `src/harpy/learning/evaluation.py` | Matched terminal rollouts, baselines, metrics, probes, criteria. |
| `src/harpy/learning/artifacts.py` | Provenance, atomic writer, strict manifest, hashes, validated loader. |
| `src/harpy/learning/bc.py` | Compact oracle examples, weighted CE, early stopping, BC artifact workflow. |
| `src/harpy/learning/ppo.py` | SB3 VecEnv/policy configuration, fresh PPO training, PPO artifact workflow. |
| `src/harpy/learning/workflows.py` | Validated cross-artifact evaluation and artifact-run orchestration. |
| `src/harpy/learning/trace.py` | Full-range hands-on episode traces and strict rendering. |
| `src/harpy/learning/cli.py` | `train-bc`, `train-ppo`, `evaluate`, and `run` commands. |
| `src/harpy/envs/sine_pitch.py` | Protected candidate-evidence seam; registered behavior remains unchanged. |
| `tests/learning/*.py` | Focused models/suites/cache/env/network/evaluation/artifact/trainer/CLI tests. |
| `tests/envs/test_sine_pitch.py` | Base evidence-seam and `_candidate_audio` regression characterization. |
| `tests/test_imports.py` | Qt/Torch/SB3 import isolation. |
| `pyproject.toml`, `uv.lock` | Optional `train` group and `harpy-sine-learn` entry point. |
| `README.md` | Hands-on install/train/evaluate/run commands and precise claims. |
| `docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md` | Engineering and actual model evidence. |

`docs/project-notebook.md` is intentionally absent from this file map because the root
checkout contains an unrelated user edit that must remain untouched.

---

### Task 1: Immutable Learning Models and Deterministic Episode Suites

**Files:**
- Create: `src/harpy/learning/__init__.py`
- Create: `src/harpy/learning/models.py`
- Create: `src/harpy/learning/suites.py`
- Create: `tests/learning/__init__.py`
- Create: `tests/learning/test_models.py`
- Create: `tests/learning/test_suites.py`

**Interfaces:**
- Produces dependency-free JSON/contract/schema types plus `ProfileName`, `TrainerKind`,
  `EvaluationSuiteId`, `DeviceName`,
  `EpisodeSpec`, `EpisodeSuite`, `BCEpisodeSplits`, `BCProfile`, `PPOProfile`, and
  immutable `PROFILE_CONFIGS`, `BCEpochMetrics`, `BCTrainingSummary`, and
  `PPOTrainingSummary`.
- Produces `fixed_evaluation_suite`, `build_bc_episode_splits`,
  `ppo_training_episode`, and `suite_digest`.
- Task 3 consumes episode lookup/reset options; Tasks 4, 6, 7, and 8 consume the
  closed profile/trainer/device models; Task 5 consumes the fixed suites.
- This task imports NumPy/Gym models only; it must not import Torch, SB3, Qt, GUI, or
  artifact code.

- [ ] **Step 1: Write immutable-model RED tests**

Create literal validation tests:

```python
def test_episode_spec_owns_only_exact_reset_truth() -> None:
    episode = EpisodeSpec(target_note_index=7, source_pitch_cents=5_432)
    assert episode.pair == (7, 5_432)
    first = episode.reset_options()
    second = episode.reset_options()
    assert first == {"target_note_index": 7, "source_pitch_cents": 5_432}
    assert first is not second


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_note_index", True), ("target_note_index", 25),
     ("source_pitch_cents", False), ("source_pitch_cents", 4_799)],
)
def test_episode_spec_rejects_invalid_integer_truth(field: str, value: object) -> None:
    values = {"target_note_index": 0, "source_pitch_cents": 4_800, field: value}
    with pytest.raises(ValueError, match=field):
        EpisodeSpec(**values)


def test_profiles_pin_rollout_aligned_values() -> None:
    smoke = PROFILE_CONFIGS[ProfileName.SMOKE]
    checkpoint = PROFILE_CONFIGS[ProfileName.CHECKPOINT]
    assert (smoke.bc.train_episodes, smoke.bc.validation_episodes) == (128, 64)
    assert (checkpoint.bc.train_episodes, checkpoint.bc.validation_episodes) == (4_096, 512)
    assert (smoke.ppo.total_timesteps, smoke.ppo.n_steps, smoke.ppo.batch_size) == (
        2_048, 256, 64
    )
    assert (
        checkpoint.ppo.total_timesteps,
        checkpoint.ppo.n_steps,
        checkpoint.ppo.batch_size,
    ) == (256_000, 1_024, 256)
    assert checkpoint.ppo.total_timesteps % checkpoint.ppo.n_steps == 0
    assert checkpoint.ppo.gamma == 1.0
```

Also pin BC epochs/patience/batch/AdamW values; PPO `n_epochs=10`, `3e-4`
learning rate, `gae_lambda=.95`, `clip_range=.2`, `ent_coef=.01`, and
`vf_coef=.5`; exact enum values; frozen/slots behavior; nonnegative integer seed
validation helpers; and no mutable defaults.

- [ ] **Step 2: Run model tests and observe RED**

```bash
uv run pytest tests/learning/test_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named
'harpy.learning'`. Do not create empty production modules before this run.

- [ ] **Step 3: Implement the immutable model surface**

Use these exact public shapes:

```python
class ProfileName(StrEnum):
    SMOKE = "smoke"
    CHECKPOINT = "checkpoint"


class TrainerKind(StrEnum):
    BC = "bc"
    PPO = "ppo"


class DeviceName(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class EvaluationSuiteId(StrEnum):
    SMOKE = "harpy-sine-policy-eval-smoke-v1"
    IID = "harpy-sine-policy-eval-iid-v1"
    REGISTER_OOD = "harpy-sine-policy-eval-register-ood-v1"


ENVIRONMENT_ID = "Harpy/SinePitch-v0"
ENVIRONMENT_CONTRACT_ID = "harpy-sine-pitch-v0-contract-v1"
SPECTRUM_GRID_ID = "harpy-sine-spectrum-grid-v1"
PREPROCESSING_SCHEMA_ID = "harpy-sine-policy-observation-v1"
ARCHITECTURE_SCHEMA_ID = "harpy-sine-policy-conv-v1"

type JSONScalar = None | bool | int | float | str
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    target_note_index: int
    source_pitch_cents: int

    @property
    def pair(self) -> tuple[int, int]: ...
    def reset_options(self) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class EpisodeSuite:
    schema_version: int
    suite_id: EvaluationSuiteId
    suite_seed: int
    episodes: tuple[EpisodeSpec, ...]
    digest_sha256: str


@dataclass(frozen=True, slots=True)
class BCEpisodeSplits:
    distribution_id: str
    run_seed: int
    profile: ProfileName
    training: tuple[EpisodeSpec, ...]
    validation: tuple[EpisodeSpec, ...]
    training_digest_sha256: str
    validation_digest_sha256: str


@dataclass(frozen=True, slots=True)
class BCEpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True, slots=True)
class BCTrainingSummary:
    history: tuple[BCEpochMetrics, ...]
    selected_epoch: int
    training_examples: int
    validation_examples: int
    training_wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class PPOTrainingSummary:
    requested_environment_steps: int
    completed_environment_steps: int
    training_wall_time_seconds: float
```

`BCProfile` and `PPOProfile` own exactly the checked-in values from the design;
`TrainingProfile` combines them and declares the smoke or IID/OOD evaluation suite.
All integer validators use `operator.index`, reject booleans, and return owned Python
integers. Summary models require finite nonnegative metrics/times, valid epoch/history
ordering, a selected epoch present in history, positive dataset counts, and exact
requested/completed PPO steps. `harpy.learning.__init__` exports only these
dependency-free models and does not import sibling modules implicitly.

- [ ] **Step 4: Run model tests GREEN**

Run `uv run pytest tests/learning/test_models.py -q` and require all tests to pass.

- [ ] **Step 5: Write fixed-suite and split RED tests**

Pin the exact fixed digests and representative membership:

```python
@pytest.mark.parametrize(
    ("suite_id", "count", "seed", "digest"),
    [
        (EvaluationSuiteId.SMOKE, 32, 202_608_100,
         "de8033b443976623b67077e84205795a02278bc53ada3611f0fed628139e75b2"),
        (EvaluationSuiteId.IID, 256, 202_608_101,
         "302be5ee0eb1556391d60646ef98aa8e7b24e104b4f54e3bf08181d1ccbdcc12"),
        (EvaluationSuiteId.REGISTER_OOD, 256, 202_608_102,
         "97358f16696601c26fd7721b10810d858bf545c6ca4bb3f82ded2e599b619710"),
    ],
)
def test_fixed_suite_identity(suite_id, count, seed, digest) -> None:
    suite = fixed_evaluation_suite(suite_id)
    assert (len(suite.episodes), suite.suite_seed, suite.digest_sha256) == (
        count, seed, digest
    )
    assert suite_digest(
        suite_id=suite.suite_id,
        suite_seed=suite.suite_seed,
        episodes=suite.episodes,
    ) == digest
```

Assert smoke target counts are 1/2; IID and OOD target counts are 10/11; OOD has
exactly 128 lower and 128 upper sources and every target's band counts differ by at
most one. Assert all pairs are unique, initial error is `>5`, domains are exact, and
all fixed suites are mutually disjoint.

Pin seed-0 BC split digests:

```python
EXPECTED_BC_DIGESTS = {
    ProfileName.SMOKE: (
        "ac43951b718486402500191dcbee7b41ef7844dfe16fbd8fb727611cba6a0833",
        "d52d911cde6b941ff329f0531ae168d2867ef06739f7ce93a4cc9439b55414f0",
    ),
    ProfileName.CHECKPOINT: (
        "f9664dc49d64b9e87e9bf5b0a0e823d7c2e78243b9c380bcbc3e10ecf5bd7a26",
        "3904604026b8738cc27801505340cc6517f0dd452cdcf8965d1926cf8f9e964f",
    ),
}
```

For each profile, assert exact train/validation length, pair uniqueness, balanced
targets, central source domain, fixed-suite exclusion, and train/validation
disjointness. For `ppo_training_episode`, assert seed/index repeatability, 25-episode
target blocks are permutations, fixed pairs are excluded, negative/bool indices fail,
and sampled pairs may repeat only across distinct indices.

- [ ] **Step 6: Run suite tests and observe RED**

```bash
uv run pytest tests/learning/test_suites.py -q
```

Expected: import failure for `harpy.learning.suites`, not a missing fixture.

- [ ] **Step 7: Implement canonical deterministic generation**

Use `SUITE_SCHEMA_VERSION = 1`,
`TRAIN_DISTRIBUTION_ID = "harpy-sine-policy-train-v1"`, and integer namespace codes:

```python
_BC_TRAIN_CODE = 201
_BC_VALIDATION_CODE = 202
_PPO_TRAIN_CODE = 203
```

For every 25-target block, construct
`default_rng(SeedSequence([1, namespace_or_suite_seed, run_seed_if_any,
block_index]))` and take a permutation of `0..24`. Fixed-suite source selection uses
`SeedSequence([1, suite_seed, episode_index, target_index, band_code])` over the sorted
remaining eligible values. BC source selection uses
`SeedSequence([1, split_code, run_seed, episode_index, target_index])`, samples without
replacement, and excludes all fixed pairs; validation additionally excludes training.
PPO uses the same balanced target blocks, source range `5000..7000`, and fixed-pair
exclusion. It selects from the sorted eligible source list with
`SeedSequence([1, 203, run_seed, episode_index, target_index])`; pairs may repeat only
across distinct episode indices.

Every single source draw is exactly
`int(default_rng(seed_sequence).choice(np.asarray(sorted_eligible, dtype=np.int64)))`.
For BC without-replacement sets, remove the accepted pair from the sorted eligible set
before the next indexed draw; do not substitute permutation, rejection sampling, or a
module-global generator.

Use `band_code=0` for central and lower draws and `band_code=1` for upper draws. OOD
alternates each target between lower and upper bands. Every target with ten occurrences
starts lower. Of the six targets with an eleventh occurrence, the first three sorted
target IDs start lower and the last three start upper, producing exactly 128/128.

`suite_digest` hashes UTF-8 canonical JSON plus one newline:

```python
{
    "schema_version": 1,
    "suite_id": suite_id.value,
    "suite_seed": suite_seed,
    "episodes": [
        {"episode_index": i, "target_note_index": e.target_note_index,
         "source_pitch_cents": e.source_pitch_cents}
        for i, e in enumerate(episodes)
    ],
}
```

Serialize with `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`. BC
digests use this exact payload, also followed by one newline:

```python
{
    "schema_version": 1,
    "distribution_id": TRAIN_DISTRIBUTION_ID,
    "split": split_name,  # exactly "training" or "validation"
    "profile": profile.value,
    "run_seed": run_seed,
    "episodes": [
        {"episode_index": i, "target_note_index": e.target_note_index,
         "source_pitch_cents": e.source_pitch_cents}
        for i, e in enumerate(episodes)
    ],
}
```

- [ ] **Step 8: Run Task 1 gates and commit**

```bash
uv run pytest tests/learning/test_models.py tests/learning/test_suites.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/learning tests/learning
git commit -m "feat: define learned-policy episode suites"
```

---

### Task 2: Backward-Compatible Candidate Evidence Seam

**Files:**
- Modify: `src/harpy/envs/sine_pitch.py:41-270`
- Modify: `tests/envs/test_sine_pitch.py`

**Interfaces:**
- Produces private `_CandidateEvidence(candidate_audio, spectrum)` and protected
  `SinePitchEnv._candidate_evidence(candidate_cents)` for Task 3.
- Keeps `_render_candidate`, `SinePitchEnv.__init__`, Gym spaces, registration, base
  `_candidate_audio`, and every public return value unchanged.
- This task does not import or mention `harpy.learning` from `harpy.envs`.

- [ ] **Step 1: Strengthen the existing base-environment characterization**

Add tests before changing production:

```python
def test_registered_base_candidate_evidence_retains_owned_audio() -> None:
    env = SinePitchEnv()
    observation, _ = env.reset(
        options={"target_note_index": 12, "source_pitch_cents": 6_000}
    )
    audio = env._candidate_audio
    spectrum = env._spectrum
    assert audio is not None and spectrum is not None
    assert audio.dtype == spectrum.dtype == np.float32
    assert audio.flags.owndata and not audio.flags.writeable
    assert spectrum.flags.owndata and not spectrum.flags.writeable
    assert not np.shares_memory(audio, observation["spectrum"])
    assert not np.shares_memory(spectrum, observation["spectrum"])
```

Retain and rerun all existing render-failure transaction, fresh-engine spy,
block/action-order identity, `_render_candidate` call-count, and `_candidate_audio`
ownership tests. Do not mention the not-yet-existing `_candidate_evidence` hook in this
pre-change characterization.

- [ ] **Step 2: Run the characterization GREEN before production changes**

```bash
uv run pytest tests/envs/test_sine_pitch.py tests/envs/test_registration.py -q
```

Expected: all tests pass on the pre-change implementation. This is a characterization
baseline, not the RED.

- [ ] **Step 3: Write the seam RED test**

```python
def test_candidate_evidence_hook_can_omit_private_audio_without_changing_observation() -> None:
    class SpectrumOnlyEnv(SinePitchEnv):
        def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
            evidence = super()._candidate_evidence(candidate_cents)
            return _CandidateEvidence(None, evidence.spectrum)

    env = SpectrumOnlyEnv()
    observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    assert env._candidate_audio is None
    assert observation["spectrum"].shape == (1_961,)
```

In the same RED slice, add a spy subclass whose `_candidate_evidence` call count will
be exactly one per reset/applied action and zero for blocked actions/Submit after the
seam exists. Before production changes this fails because there is no hook to override.

- [ ] **Step 4: Run the seam test and observe RED**

Run its exact node ID. Expected: `_CandidateEvidence` or `_candidate_evidence` is
absent, or `_observation` rejects the intentional `None` audio.

- [ ] **Step 5: Implement the protected evidence seam minimally**

Add:

```python
@dataclass(frozen=True, slots=True, eq=False)
class _CandidateEvidence:
    candidate_audio: np.ndarray | None
    spectrum: np.ndarray


def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
    audio, spectrum = self._render_candidate(candidate_cents)
    return _CandidateEvidence(audio, spectrum)
```

Have reset and applied-action paths build evidence in locals, then atomically commit
`_candidate_audio` and `_spectrum` with all other episode state. `_observation` requires
initialized source, target, and spectrum but not universal audio. Do not alter
`_render_candidate` or its return type. The registered base always calls this default
hook and therefore still owns audio.

- [ ] **Step 6: Run Task 2 gates and commit**

```bash
uv run pytest tests/envs/test_sine_pitch.py tests/envs/test_registration.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/envs/sine_pitch.py tests/envs/test_sine_pitch.py
git commit -m "refactor: expose sine candidate evidence seam"
```

---

### Task 3: Bounded Spectrum Cache and Learning Environments

**Files:**
- Create: `src/harpy/learning/cache.py`
- Create: `src/harpy/learning/envs.py`
- Create: `tests/learning/test_cache.py`
- Create: `tests/learning/test_envs.py`

**Interfaces:**
- Consumes Task 1 `EpisodeSpec`/PPO indexed sequence and Task 2
  `_CandidateEvidence` hook.
- Produces `SpectrumEvidenceCache.get_or_create`, `SpectrumEvidenceProvider`,
  `CachedSinePitchEnv`, `ScheduledEpisodeEnv`, `make_cached_sine_pitch_env`, and
  `make_ppo_training_env`.
- Task 5 uses cached factories for matched evaluation; Tasks 7/8 use the same shared
  cache for training.

- [ ] **Step 1: Write cache RED tests with an injected miss renderer**

```python
def test_cache_miss_then_hit_owns_isolated_spectra() -> None:
    calls: list[int] = []
    cache = SpectrumEvidenceCache()

    def render() -> np.ndarray:
        calls.append(6_000)
        return np.linspace(0.0, 1.0, 1_961, dtype=np.float32)

    first = cache.get_or_create(6_000, on_miss=render)
    second = cache.get_or_create(6_000, on_miss=lambda: pytest.fail("unexpected miss"))
    assert calls == [6_000]
    assert np.array_equal(first, second)
    assert not np.shares_memory(first, second)
    assert not first.flags.writeable and not second.flags.writeable
    assert len(cache) == 1
```

Also test bool/noninteger/out-of-range keys, wrong dtype/shape/rank/nonfinite miss
values, miss exceptions leaving no entry, C-contiguous conversion, 9,801 exact keys,
and a mutation probe that deliberately alters a private cache entry and requires the
next lookup to raise a digest-integrity error.

- [ ] **Step 2: Run cache tests and observe RED**

Run `uv run pytest tests/learning/test_cache.py -q`; expected missing module.

- [ ] **Step 3: Implement the cache**

Use:

```python
class SpectrumEvidenceCache:
    def get_or_create(
        self,
        candidate_pitch_cents: object,
        *,
        on_miss: Callable[[], np.ndarray],
    ) -> np.ndarray: ...

    def __len__(self) -> int: ...
    @property
    def spectrum_bytes(self) -> int: ...
```

Store private `_CacheEntry(spectrum, sha256)` values. Validate keys with
`operator.index`, explicitly reject bool, and require `1100..10900`. On a miss,
validate and copy into an owned C-contiguous float32 array, mark it read-only, hash its
raw bytes/shape/dtype, then commit the mapping. On every hit, verify the stored digest
before returning another owned read-only copy. `spectrum_bytes` reaches exactly
`76_879_044` at 9,801 full entries; mapping overhead is not included.

- [ ] **Step 4: Run cache tests GREEN**

Run the focused cache file and require all tests to pass.

- [ ] **Step 5: Write cached/scheduled environment RED tests**

Cover the actual synth path:

```python
@pytest.mark.parametrize("candidate_cents", [1_100, 4_800, 6_000, 10_900])
def test_cached_miss_and_hit_are_bit_identical_to_direct(candidate_cents: int) -> None:
    options = {"target_note_index": 12, "source_pitch_cents": 6_000}
    direct = SinePitchEnv()
    cached = CachedSinePitchEnv(SpectrumEvidenceProvider(SpectrumEvidenceCache()))
    direct.reset(options=options)
    cached.reset(options=options)
    direct_evidence = direct._candidate_evidence(candidate_cents).spectrum
    miss = cached._candidate_evidence(candidate_cents).spectrum
    hit = cached._candidate_evidence(candidate_cents).spectrum
    assert np.array_equal(direct_evidence.view(np.uint32), miss.view(np.uint32))
    assert np.array_equal(miss.view(np.uint32), hit.view(np.uint32))
```

Add real reset/step trajectories proving equal observations/rewards/info/flags/results
for inverse actions and two sequences reaching the same controls. Assert the cached
subclass retains `_candidate_audio is None`; base env retains audio. Assert cache,
env-state spectrum, and returned observation never share memory.

For `ScheduledEpisodeEnv`, assert exact indexed pair order, no metadata in observation
or info, forwarded seed, `{}` accepted as no external override, nonempty external
options rejected, and the index advances only after a successful underlying reset.

- [ ] **Step 6: Run environment tests and observe RED**

```bash
uv run pytest tests/learning/test_envs.py -q
```

Expected: missing `harpy.learning.envs` interfaces.

- [ ] **Step 7: Implement cached and scheduled environments**

Use:

```python
class SpectrumEvidenceProvider:
    def __init__(self, cache: SpectrumEvidenceCache) -> None: ...
    def spectrum_for_cents(self, effective_pitch_cents: int) -> np.ndarray: ...


class CachedSinePitchEnv(SinePitchEnv):
    def __init__(
        self,
        evidence_provider: SpectrumEvidenceProvider,
        observation_mode: ObservationMode = ObservationMode.SPECTRUM,
    ) -> None: ...

    def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence: ...


class ScheduledEpisodeEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, episode_at: Callable[[int], EpisodeSpec]) -> None: ...
    @property
    def next_episode_index(self) -> int: ...
    def reset(self, *, seed: int | None = None,
              options: dict[str, object] | None = None) -> tuple[Observation, dict[str, object]]: ...


def make_cached_sine_pitch_env(
    cache: SpectrumEvidenceCache,
    *,
    observation_mode: ObservationMode = ObservationMode.SPECTRUM,
) -> CachedSinePitchEnv: ...


def make_ppo_training_env(
    *,
    run_seed: int,
    cache: SpectrumEvidenceCache,
) -> ScheduledEpisodeEnv: ...
```

`SpectrumEvidenceProvider` owns one private direct `SinePitchEnv` renderer. Its miss
callback calls that renderer's protected `_candidate_evidence` seam, passes only the
spectrum to the cache, and discards audio. Its public method always returns a fresh
cache-isolated spectrum. `CachedSinePitchEnv._candidate_evidence` delegates to that
provider and returns `_CandidateEvidence(None, spectrum)`. This keeps exact rendering
and cache population out of the BC dataset and avoids a second spectrum formula.

The scheduled wrapper calls the real base `reset` with a fresh
`EpisodeSpec.reset_options()` dict and implements no step/reward logic.

`make_cached_sine_pitch_env` constructs a provider around the caller-owned cache, so
multiple zero-argument factories written as
`lambda: make_cached_sine_pitch_env(shared_cache)` share evidence but never env state.
Test two such env instances: the second reuses the first's cache entry while reset,
controls, candidate spectrum ownership, and close lifecycle remain independent.

`make_ppo_training_env(run_seed=..., cache=...)` wraps a cached spectrum env with
`episode_at=lambda index: ppo_training_episode(run_seed=run_seed,
episode_index=index)` and returns the scheduled Gym env expected by Task 8's
`make_ppo_vec_env(lambda: make_ppo_training_env(...))`; it never registers a Gym ID.

- [ ] **Step 8: Add the exhaustive reachable-cache contract**

One marked integration test fills all integer keys `1100..10900` through the real
renderer, asserts every spectrum is finite/shape-stable, `len(cache)==9801`, and exact
`spectrum_bytes`. Keep representative direct/cache bit tests separate so a cache bug
cannot make both sides share the same implementation path.

- [ ] **Step 9: Run Task 3 gates and commit**

```bash
uv run pytest tests/learning/test_cache.py tests/learning/test_envs.py \
  tests/envs/test_sine_pitch.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/learning/cache.py src/harpy/learning/envs.py \
  tests/learning/test_cache.py tests/learning/test_envs.py
git commit -m "feat: cache learned-policy spectrum evidence"
```

---

### Task 4: Optional Training Stack, Observation Adapter, and Shared Network

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/harpy/learning/errors.py`
- Create: `src/harpy/learning/dependencies.py`
- Create: `src/harpy/learning/observations.py`
- Create: `src/harpy/learning/network.py`
- Create: `src/harpy/learning/actors.py`
- Create: `tests/learning/test_import_boundaries.py`
- Create: `tests/learning/test_observations.py`
- Create: `tests/learning/test_network.py`
- Create: `tests/learning/test_actors.py`
- Modify: `tests/test_imports.py`

**Interfaces:**
- Produces lazy `require_training_dependencies`, exact raw-to-policy
  `preprocess_observation`, `PolicyObservationWrapper`, shared `SineFeatureEncoder`,
  `BCPolicyNetwork`, SB3 `HarpySineFeaturesExtractor`, and narrow `Actor`, `BCActor`,
  and `PPOActor`.
- Tasks 7/8 consume the network and actor adapters; Task 5 consumes only the
  dependency-free `Actor` protocol via `TYPE_CHECKING`/structural typing.
- `harpy.learning.__init__` remains heavy-dependency-free even after this task.

- [ ] **Step 1: Write optional-dependency/import RED tests**

Pin the exact contract:

```python
def test_learning_import_loads_no_training_or_qt_modules() -> None:
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys, harpy.learning; "
            "assert 'torch' not in sys.modules; "
            "assert 'stable_baselines3' not in sys.modules; "
            "assert not any(n == 'PySide6' or n.startswith('PySide6.') "
            "for n in sys.modules)"
        )],
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()
```

Inject an `import_module` callable that raises `ModuleNotFoundError` for `torch` and
for `stable_baselines3`; both cases must raise `DependencyUnavailableError` whose
message contains exactly `uv sync --group train`. With real installed dependencies,
assert reported versions match module `__version__` values. Extend `tests/test_imports`
so ordinary top-level, synth, env, analysis, and GUI imports do not load Torch/SB3.

- [ ] **Step 2: Run import tests and observe RED**

```bash
uv run pytest tests/learning/test_import_boundaries.py tests/test_imports.py -q
```

Expected: missing errors/dependencies modules or missing train-group metadata.

- [ ] **Step 3: Add the optional dependency group with uv**

```bash
uv add --group train 'stable-baselines3==2.9.0' 'torch>=2.8,<3'
uv lock --check
```

Do not use SB3's `[extra]` bundle. `pyproject.toml` must contain only those two direct
`train` entries. Implement:

```python
TRAIN_INSTALL_INSTRUCTION = "uv sync --group train"

class LearningContractError(ValueError): ...
class DependencyUnavailableError(RuntimeError): ...
class ArtifactError(RuntimeError): ...
class LearningExecutionError(RuntimeError): ...

@dataclass(frozen=True, slots=True)
class TrainingStack:
    torch: ModuleType
    stable_baselines3: ModuleType
    torch_version: str
    stable_baselines3_version: str

def require_training_dependencies(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> TrainingStack: ...
```

`dependencies.py` imports neither heavy package at module import time.

- [ ] **Step 4: Run dependency/import tests GREEN**

Run the focused files. Also run:

```bash
uv run python -c "import sys, harpy.learning; assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules"
```

- [ ] **Step 5: Write observation-adapter RED tests**

Build real valid raw observations from `SinePitchEnv.reset(options=...)` and pin:

```python
def test_preprocessing_has_exact_two_box_values() -> None:
    raw = {
        "spectrum": np.linspace(0, 1, 1_961, dtype=np.float32),
        "target_note": np.int64(24),
        "controls": np.array([-2, 12, -100], dtype=np.int16),
        "steps_remaining": np.int64(32),
    }
    policy = preprocess_observation(raw)
    assert tuple(policy) == ("spectrum", "state")
    np.testing.assert_array_equal(
        policy["state"], np.array([1.0, -1.0, 1.0, -1.0, 0.5], dtype=np.float32)
    )
    assert policy["spectrum"].dtype == np.float32
    assert not np.shares_memory(policy["spectrum"], raw["spectrum"])
    assert POLICY_OBSERVATION_SPACE.contains(policy)
```

Test target `(index-12)/12`, controls `/[2,12,100]`, remaining `/64`, exact key order,
fresh C-contiguous ownership, and no clamping. Reject missing/extra keys; Python ints in
place of required `np.int64`; wrong control/spectrum dtype/rank/shape; all out-of-bound
values; NaN/Inf; and oracle/evaluator aliases such as `current_pitch_coordinate`,
`source_pitch_cents`, or `final_absolute_error_cents`.

- [ ] **Step 6: Run observation tests RED, then implement the adapter**

Expected RED: `harpy.learning.observations` missing. Implement:

```python
type PolicyObservation = dict[str, np.ndarray]

POLICY_OBSERVATION_SPACE = gymnasium.spaces.Dict({
    "spectrum": gymnasium.spaces.Box(0.0, 1.0, (1_961,), np.float32),
    "state": gymnasium.spaces.Box(
        low=np.array([-1, -1, -1, -1, 0], dtype=np.float32),
        high=np.ones(5, dtype=np.float32),
        dtype=np.float32,
    ),
})

def preprocess_observation(observation: Mapping[str, object]) -> PolicyObservation: ...

class PolicyObservationWrapper(gymnasium.ObservationWrapper):
    observation_space = POLICY_OBSERVATION_SPACE
    def observation(self, observation: Mapping[str, object]) -> PolicyObservation: ...
```

Validate the frozen raw contract before constructing output. Do not rely on
`gymnasium.Space.contains` as the only diagnostic.

- [ ] **Step 7: Write exact network/actor RED tests**

Assert module topology, seeded finite batch output, and exact counts:

```python
def test_network_parameter_counts_are_versioned() -> None:
    encoder = SineFeatureEncoder()
    bc = BCPolicyNetwork()
    assert sum(p.numel() for p in encoder.parameters()) == 74_784
    assert sum(p.numel() for p in bc.parameters()) == 75_687


def test_bc_actor_uses_first_argmax_on_a_tie() -> None:
    model = ConstantLogitBC(torch.zeros(7))
    actor = BCActor(model, device=torch.device("cpu"))
    assert actor.act(valid_raw_observation()) is PitchAction.OCTAVE_DOWN
```

Inspect the two Conv layers, ReLUs, adaptive pool 16, scalar `5→32→32`, combined
`544→128`, and `128→7` head. Assert encoder output `(batch,128)`, BC logits
`(batch,7)`, float finiteness, CPU ownership, and deterministic seeded state dict.
`PPOActor` must call `predict(transformed, deterministic=True)`, validate a scalar
integer action `0..6`, and accept SB3's real unvectorized discrete result only when it
is an integral zero-dimensional NumPy array (convert with `.item()`). Reject
non-scalar arrays, floats, bools, and out-of-range results. Include a regression using
an actual SB3 prediction, not only a fake. Learned
actor signatures accept observation only—no reward, info, or result.

- [ ] **Step 8: Implement the shared network and actors**

Use:

```python
class SineFeatureEncoder(torch.nn.Module):
    output_dim: ClassVar[int] = 128
    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

class BCPolicyNetwork(torch.nn.Module):
    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

class HarpySineFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict) -> None: ...
    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

@runtime_checkable
class Actor(Protocol):
    def act(self, observation: Mapping[str, object]) -> PitchAction: ...
```

The spectrum branch is Conv1d `(1→16,k9,s4)`, ReLU, Conv1d
`(16→32,k7,s4)`, ReLU, adaptive average pool to 16. The state branch is
`5→32→32` with ReLU after each layer. Concatenate 512+32, apply `544→128` and
ReLU. `BCPolicyNetwork` adds only `128→7`. Both actors call
`preprocess_observation` themselves so inference and training share one validator.
The observation/network modules import `PREPROCESSING_SCHEMA_ID` and
`ARCHITECTURE_SCHEMA_ID` from Task 1; they do not redeclare those strings.

- [ ] **Step 9: Run Task 4 gates and commit**

```bash
uv sync --locked --group train
uv run pytest tests/learning/test_import_boundaries.py \
  tests/learning/test_observations.py tests/learning/test_network.py \
  tests/learning/test_actors.py tests/test_imports.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add pyproject.toml uv.lock src/harpy/learning tests/learning tests/test_imports.py
git commit -m "feat: add learned sine policy network"
```

---

### Task 5: Matched Terminal Evaluator and Scientific Criteria

**Files:**
- Create: `src/harpy/learning/evaluation.py`
- Create: `tests/learning/test_evaluation.py`
- Create: `tests/learning/test_criteria.py`

**Interfaces:**
- Consumes Task 1 suites, Task 3 cached env factory, Task 4 narrow `Actor`, and existing
  Milestone C baseline policies/planners.
- Produces immutable `TerminalEpisodeRecord`, `AggregateMetrics`, `EvaluationRow`,
  `EvaluationFile`, `BCScientificCriterion`, `PPOScientificCriterion`, strict codecs,
  learned/baseline rollout functions, aggregation, perturbation probes, and exact gate
  functions.
- Tasks 7/8 use evaluator output for trainer-local artifacts. Task 9 serializes these
  models and must not recompute a metric or criterion.

- [ ] **Step 1: Write learned-rollout leakage RED tests**

Use a spy actor and a one/two-episode suite:

```python
class ObservationOnlySpy:
    def __init__(self) -> None:
        self.keys: list[tuple[str, ...]] = []
    def act(self, observation: Mapping[str, object]) -> PitchAction:
        self.keys.append(tuple(sorted(observation)))
        return PitchAction.SUBMIT


def test_learned_rollout_passes_only_raw_actor_observation() -> None:
    actor = ObservationOnlySpy()
    records = evaluate_learned_actor(actor, one_episode_suite(), environment_factory=env_factory)
    assert actor.keys == [("controls", "spectrum", "steps_remaining", "target_note")]
    assert records[0].terminal_reason is TerminalReason.SUBMITTED_FAILURE
```

Poison `episode_result` before done and assert the evaluator never reads it. After done,
copy exact terminal values into an immutable record. Assert the same ordered
`EpisodeSpec` sequence/digest reaches every learned subject and that envs close after
success/failure. Reject a terminal `EpisodeResult` whose source cents or target index
does not exactly match the injected `EpisodeSpec`.

- [ ] **Step 2: Write baseline-capability and Random-seed RED tests**

Evaluate Random, Reward Search, Spectrum Peak, and Oracle on the same injected suite.
Spy on Random's RNG construction and require exactly:

```python
np.random.SeedSequence([suite.suite_seed, 1, episode_index])
```

Assert a fresh policy per episode; planner adapters own independent action cursors;
Reward Search alone receives prior reward/info; learned/Spectrum Peak/Oracle adapters
do not gain it. Keep each declared environment/observation mode in its row and never
derive an ID from unregistered cached `env.spec`.

- [ ] **Step 3: Run evaluator tests and observe RED**

```bash
uv run pytest tests/learning/test_evaluation.py -q
```

Expected: missing `harpy.learning.evaluation`.

- [ ] **Step 4: Implement terminal records and matched rollout functions**

Use:

```python
@dataclass(frozen=True, slots=True)
class TerminalEpisodeRecord:
    episode_index: int
    episode: EpisodeSpec
    submitted_success: bool
    within_5_cents: bool
    within_1_cent: bool
    final_absolute_error_cents: int
    action_count: int
    excess_actions: int | None
    invalid_action_count: int
    total_return: float
    terminal_reason: TerminalReason

@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    episodes: int
    submitted_success_rate: float
    submitted_within_1_cent_rate: float
    final_within_5_cents_rate: float
    final_within_1_cent_rate: float
    mean_absolute_final_error_cents: float
    median_absolute_final_error_cents: float
    mean_actions: float
    mean_successful_excess_actions: float | None
    mean_return: float
    truncation_rate: float
    invalid_action_rate: float

def evaluate_learned_actor(
    actor: Actor,
    suite: EpisodeSuite,
    *,
    environment_factory: Callable[[], gymnasium.Env],
) -> tuple[TerminalEpisodeRecord, ...]: ...

def evaluate_baseline_suite(
    kind: BaselineKind,
    suite: EpisodeSuite,
    *,
    cache: SpectrumEvidenceCache,
) -> tuple[TerminalEpisodeRecord, ...]: ...
```

Read `EpisodeResult` immediately after raw Gym done and before any SB3 auto-reset.
Aggregate invalid rate as `sum(invalid_action_count)/sum(action_count)`. Serialize no
NaN; successful-excess mean is `None` with no successes. OOD row production returns
lower, upper, combined in that exact order.

- [ ] **Step 5: Write metric/criteria RED tests with hand-built records**

Pin all denominators, median, `None`, and finite values. Then test:

```python
def test_bc_gate_requires_both_declared_thresholds() -> None:
    assert evaluate_bc_criterion(
        heldout_next_action_accuracy=.90,
        iid_row=row(submitted_success_rate=.75),
        eligible=True,
    ).criterion_met is True
    assert evaluate_bc_criterion(
        heldout_next_action_accuracy=.8999,
        iid_row=row(submitted_success_rate=.75),
        eligible=True,
    ).criterion_met is False


def test_ppo_gate_uses_exact_five_seed_median_and_strict_random_beats() -> None:
    rates = [.51, .52, .53, .54, .55]
    result = evaluate_ppo_criterion(
        iid_rows=tuple(ppo_row(seed=i, rate=rate) for i, rate in enumerate(rates)),
        random_iid_row=row(submitted_success_rate=.50),
        eligible=True,
    )
    assert result.median_iid_submitted_success_rate == .53
    assert result.seeds_strictly_beating_random == 5
    assert result.criterion_met is True
```

Reject missing/duplicate/extra seeds, mismatched suite/digest, and best-seed pooling.
An explicit `eligible=False` produces an ineligible result without computation.
Manifest/profile/device/source eligibility is derived and tested at Task 9's workflow
boundary, where that provenance exists; Task 5 must not infer it from an untrusted
boolean or fields it does not own. Equality with Random does not count. OOD,
perturbation, Reward Search, Spectrum Peak, and Oracle rows never enter the gates.

- [ ] **Step 6: Implement rows, probes, and criteria**

Use the exact immutable report boundary:

```python
@dataclass(frozen=True, slots=True)
class EvaluationRow:
    actor_id: str
    trainer: TrainerKind | None
    seed: int | None
    environment_id: str
    observation_mode: ObservationMode
    suite_id: EvaluationSuiteId
    suite_digest_sha256: str
    subset: str
    probe: str | None
    parameter_count: int | None
    training_environment_steps: int | None
    training_examples: int | None
    training_wall_time_seconds: float | None
    metrics: AggregateMetrics
    episodes: tuple[TerminalEpisodeRecord, ...]


@dataclass(frozen=True, slots=True)
class BCScientificCriterion:
    eligible: bool
    heldout_next_action_accuracy: float | None
    iid_submitted_success_rate: float | None
    criterion_met: bool | None
    status: str


@dataclass(frozen=True, slots=True)
class PPOScientificCriterion:
    eligible: bool
    median_iid_submitted_success_rate: float | None
    seeds_strictly_beating_random: int | None
    criterion_met: bool | None
    status: str


@dataclass(frozen=True, slots=True)
class EvaluationFile:
    schema_version: int
    suite_id: EvaluationSuiteId
    suite_digest_sha256: str
    rows: tuple[EvaluationRow, ...]
    next_action_accuracy: float | None

    def to_document(self) -> dict[str, JSONValue]: ...

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> EvaluationFile: ...
```

`subset` is exactly `combined`, `lower`, or `upper`; IID/smoke use `combined`, and OOD
rows are emitted lower, upper, combined in that order. Zero-spectrum and
deterministically shuffled-spectrum wrappers alter only a fresh spectrum copy and
label `probe` as `zero_spectrum` or `shuffled_spectrum`; they are sensitivity
diagnostics only.

Pin shuffled evidence as `harpy-sine-spectrum-shuffle-v1`: construct one permutation
with `default_rng(SeedSequence([202_608_103, 1])).permutation(1_961)` and reuse that
same immutable permutation for every timestep, episode, actor, and suite. Tests prove
repeatability, actor-order independence, an unchanged scalar branch and spectrum
multiset, changed bin order for nonconstant evidence, and exclusion from both gates.
Smoke actors run both probes on the smoke suite; checkpoint BC/PPO actors run both
probes on IID only. OOD remains the unperturbed register-generalization report.

`EvaluationFile` schema version 1 is the exact trainer-artifact payload. Its strict
codec rejects duplicate/missing/extra fields, malformed enums, suite/row mismatch,
nonfinite metrics, bad denominators, reordered/wrong episode membership, and a
next-action accuracy outside `[0,1]`. Only BC IID/checkpoint (and optional BC smoke
diagnostic) may set `next_action_accuracy`; PPO and OOD use `None`.

`BCScientificCriterion` and `PPOScientificCriterion` explicitly distinguish
`eligible`, `criterion_met`, and `criterion_not_met`/`ineligible` status strings.
Input validation precedes computation.

- [ ] **Step 7: Run Task 5 gates and commit**

```bash
uv run pytest tests/learning/test_evaluation.py tests/learning/test_criteria.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/learning/evaluation.py tests/learning/test_evaluation.py \
  tests/learning/test_criteria.py
git commit -m "feat: evaluate learned sine policies"
```

---

### Task 6: Strict Local Artifact Lifecycle

**Files:**
- Create: `src/harpy/learning/artifacts.py`
- Create: `tests/learning/test_artifacts.py`

**Interfaces:**
- Consumes Task 1 trainer/profile/device enums and dependency-free stable schema IDs.
- Produces strict manifest/provenance models, `ArtifactWriter`, `LoadedArtifact`, a
  writer-only `PendingArtifactView`, canonical JSON, create-only report writing,
  source capture, inventory validation, and hash-checked loading.
- Tasks 7/8 publish trainer-specific models through callbacks. Task 9 loads only
  `LoadedArtifact`; it never bypasses validation.

- [ ] **Step 1: Write strict manifest/provenance RED tests**

Pin schema-v1 rejection of duplicate/missing/extra keys, bool-as-int, invalid enums,
unsafe/absolute/parent paths, malformed hashes, wrong sizes, invalid timestamps, and
NaN/Infinity. Use temporary Git repos to assert clean, tracked-dirty, staged, and
untracked status; SHA-256 of `uv.lock`; SHA-256 of tracked diff; and
`required_inputs_committed` only when the suite/config/lock paths are tracked at HEAD.

For every JSON payload filename in the closed inventory, also rewrite a valid artifact
with duplicate keys, malformed syntax, NaN, or Infinity, then recompute its size/hash
and the manifest so integrity checks alone pass. `load_artifact` must still reject the
payload through the duplicate-detecting finite JSON decoder before actor construction.

Required closed inventories are:

```python
EXPECTED_INVENTORY = {
    (TrainerKind.BC, ProfileName.SMOKE): (
        "training-config.json", "training-summary.json", "model.pt", "evaluation-smoke.json"
    ),
    (TrainerKind.PPO, ProfileName.SMOKE): (
        "training-config.json", "training-summary.json", "model.zip", "evaluation-smoke.json"
    ),
    (TrainerKind.BC, ProfileName.CHECKPOINT): (
        "training-config.json", "training-summary.json", "model.pt",
        "evaluation-iid.json", "evaluation-ood.json"
    ),
    (TrainerKind.PPO, ProfileName.CHECKPOINT): (
        "training-config.json", "training-summary.json", "model.zip",
        "evaluation-iid.json", "evaluation-ood.json"
    ),
}
```

Manifest never hashes itself. Complete loaders reject temporary, missing, unexpected,
wrong-size, or wrong-hash files before any model constructor runs.
Public `load_artifact` always rejects an incomplete manifest. A separate writer-owned
pending view is the only pre-completion input permitted for the required persisted
model reload.

- [ ] **Step 2: Run artifact tests and observe RED**

Run `uv run pytest tests/learning/test_artifacts.py -q`; expected missing module.

- [ ] **Step 3: Implement strict immutable artifact records**

Define:

```python
ARTIFACT_SCHEMA_VERSION = 1

class ArtifactStatus(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"

class CriterionStatus(StrEnum):
    INELIGIBLE = "ineligible"
    ELIGIBLE_FOR_AGGREGATE = "eligible_for_aggregate"
    CRITERION_MET = "criterion_met"
    CRITERION_NOT_MET = "criterion_not_met"

@dataclass(frozen=True, slots=True)
class FileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True, slots=True)
class SourceStatus:
    commit: str
    dirty_tree: bool
    tracked_diff_sha256: str
    dependency_lock_sha256: str
    required_inputs_committed: bool

@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    python_version: str
    platform: str
    processor: str
    numpy_version: str
    gymnasium_version: str
    torch_version: str
    stable_baselines3_version: str
    device: DeviceName
    device_description: str
    cuda_runtime_version: str | None
    cuda_driver_version: str | None

@dataclass(frozen=True, slots=True)
class BCTrainingCounts:
    configured_training_episodes: int
    configured_validation_episodes: int
    training_examples: int | None
    validation_examples: int | None

@dataclass(frozen=True, slots=True)
class PPOTrainingCounts:
    requested_environment_steps: int
    completed_environment_steps: int | None

type TrainingCounts = BCTrainingCounts | PPOTrainingCounts

@dataclass(frozen=True, slots=True)
class TrainingConfigDocument:
    schema_version: int
    trainer: TrainerKind
    profile: ProfileName
    seed: int
    device: DeviceName
    environment_id: str
    environment_contract_id: str
    train_distribution_id: str
    evaluation_suites: tuple[tuple[EvaluationSuiteId, str], ...]
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    profile_config: BCProfile | PPOProfile
    bc_training_digest_sha256: str | None
    bc_validation_digest_sha256: str | None
    def to_document(self) -> dict[str, JSONValue]: ...
    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> TrainingConfigDocument: ...

@dataclass(frozen=True, slots=True)
class TrainingSummaryDocument:
    schema_version: int
    trainer: TrainerKind
    profile: ProfileName
    seed: int
    summary: BCTrainingSummary | PPOTrainingSummary
    def to_document(self) -> dict[str, JSONValue]: ...
    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> TrainingSummaryDocument: ...

@dataclass(frozen=True, slots=True)
class ArtifactCompletion:
    completed_at_utc: str
    training_counts: TrainingCounts
    evaluation_device: DeviceName
    bc_criterion_met: bool | None

@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: int
    status: ArtifactStatus
    trainer: TrainerKind
    profile: ProfileName
    seed: int
    created_at_utc: str
    completed_at_utc: str | None
    source: SourceStatus
    runtime: RuntimeStatus
    environment_id: str
    environment_contract_id: str
    train_distribution_id: str
    evaluation_suites: tuple[tuple[str, str], ...]
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    parameter_count: int
    training_counts: TrainingCounts
    evaluation_device: DeviceName | None
    criterion_eligible: bool
    criterion_status: CriterionStatus
    criterion_met: bool | None
    files: tuple[FileRecord, ...]
```

Use a duplicate-detecting JSON decoder and explicit field codecs. Canonical JSON is
UTF-8, sorted compact keys, `allow_nan=False`, and exactly one newline.

Import Task 1's recursive JSON vocabulary rather than using `Any`. Every float codec
rejects NaN and Infinity. `criterion_met` is `None` for ineligible artifacts and PPO
artifacts that are merely eligible for the later aggregate, otherwise it is the actual
single-artifact BC result; PPO artifacts never pretend to own the five-seed aggregate
result.

The manifest's count variant must match its trainer. An incomplete BC manifest pins
configured episode counts and has `training_examples=None`/
`validation_examples=None`; an incomplete PPO manifest pins requested steps and has
`completed_environment_steps=None`. Completion supplies the actual BC example counts
or exact PPO completed steps and cross-validates them against `training-summary.json`
and the checked-in profile.

`TrainingConfigDocument` and `TrainingSummaryDocument` have explicit duplicate-
detecting `to_document`/`from_document` codecs. Their trainer/profile/seed, contract
IDs, suite IDs/digests, profile config, and counts must match each other and the
manifest. BC alone carries the two actual split digests; PPO requires both digest
fields to be `None`. The summary union must match `trainer`, and all history/timing/
count numbers must be finite and validated by Task 1's immutable records.

- [ ] **Step 4: Write bootstrap/publication RED tests**

Inject failures at parent creation, exclusive directory creation, incomplete-manifest
temp write/flush/rename, JSON/model saver, payload rename, hash calculation, and final
manifest publication. Raise `KeyboardInterrupt` before and after first manifest.
Assert:

- existing output paths remain untouched;
- handled pre-manifest failure removes its empty/temp-only directory;
- post-bootstrap failure leaves a valid atomic incomplete manifest;
- completion happens only with exact inventory and hashes;
- a writer pending view validates only already-published requested payloads, while the
  public disk loader still rejects the incomplete artifact;
- completion preserves every bootstrap field, cross-validates trainer-specific counts,
  and derives file records/eligibility/status instead of trusting the caller;
- criterion failure still permits complete status;
- `write_new_bytes` is atomic create-only and never overwrites;
- an uncatchable empty residue gets the specific safe-to-remove error.

- [ ] **Step 5: Implement writer/loader/create-only primitives**

Use:

```python
@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    root: Path
    manifest: ArtifactManifest
    def file(self, relative_path: str) -> Path: ...
    def document(self, relative_path: str) -> dict[str, JSONValue]: ...

@dataclass(frozen=True, slots=True)
class PendingArtifactView:
    root: Path
    manifest: ArtifactManifest
    files: tuple[FileRecord, ...]
    def file(self, relative_path: str) -> Path: ...
    def document(self, relative_path: str) -> dict[str, JSONValue]: ...

@dataclass(slots=True)
class ArtifactWriter:
    @classmethod
    def begin(cls, output: Path, manifest: ArtifactManifest) -> ArtifactWriter: ...
    @property
    def file_records(self) -> tuple[FileRecord, ...]: ...
    def publish_json(self, filename: str, document: Mapping[str, JSONValue]) -> FileRecord: ...
    def publish_model(self, filename: str, save: Callable[[Path], None]) -> FileRecord: ...
    def pending_view(self, required_names: Sequence[str]) -> PendingArtifactView: ...
    def complete(self, completion: ArtifactCompletion) -> LoadedArtifact: ...

def canonical_json_bytes(document: Mapping[str, JSONValue]) -> bytes: ...
def decode_json_bytes(content: bytes) -> dict[str, JSONValue]: ...
def capture_source_status(start: Path) -> SourceStatus: ...
def required_payload_names(trainer: TrainerKind, profile: ProfileName) -> tuple[str, ...]: ...
def read_json_document(path: Path) -> dict[str, JSONValue]: ...
def read_training_config(
    artifact: LoadedArtifact | PendingArtifactView,
) -> TrainingConfigDocument: ...
def read_training_summary(
    artifact: LoadedArtifact | PendingArtifactView,
) -> TrainingSummaryDocument: ...
def load_artifact(root: Path) -> LoadedArtifact: ...
def write_new_bytes(path: Path, content: bytes) -> None: ...
```

Mask/defer handled SIGINT only around exclusive mkdir plus first atomic incomplete
manifest. Flush and `fsync` files before `os.replace`. For create-only report files,
publish a flushed sibling temporary with atomic no-clobber linking; never use a
check-then-replace race. Resolve artifact paths before comparison, and never follow a
manifest relative path outside its root. Model temporary names preserve the final
`.pt` or `.zip` suffix so Torch and SB3 write the exact requested file instead of
silently appending another extension.

`pending_view` is not a disk loader: only the live `ArtifactWriter` can create it. It
requires `status="incomplete"`, verifies the requested files are already published in
the writer's in-memory records, then rechecks size/hash before returning. It permits a
trainer-specific model reload but not public `evaluate`, public `run`, completion
claims, or direct artifact consumption. The owning train workflow may use the returned
actor for its required internal evaluation. Tests prove `load_artifact` still rejects
the same on-disk incomplete directory before any actor constructor runs.

`ArtifactWriter.complete` derives the final manifest rather than accepting one from a
caller. All bootstrap identity/provenance/runtime/contract/profile/seed/created fields
must remain byte/value equal. Only completion time, actual trainer-specific counts,
evaluation device, derived criterion fields, and the writer-derived closed file records
may change. Mutation tests try to replace every protected bootstrap field and must fail
without publishing a complete manifest.

The writer derives eligibility from profile, declared seed, clean committed source,
CPU training, and CPU evaluation. Cross-field status is closed: ineligible artifacts
use `INELIGIBLE`/`criterion_met=None`; eligible PPO seeds use
`ELIGIBLE_FOR_AGGREGATE`/`None`; eligible BC uses `CRITERION_MET`/`True` or
`CRITERION_NOT_MET`/`False`. The caller cannot self-declare eligibility or status.

Scientific eligibility is true only for checkpoint profile, declared trainer seed,
CPU, clean source, and committed required inputs. Dirty/CUDA/extra-seed runs remain
valid complete exploratory artifacts.

`start` is always an anchor inside the executing package worktree (the resolved
`artifacts.py`/`workflows.py` source path), and Git discovers that anchor's worktree
root. It is never the process CWD or output path. Tests run clean code from one linked
worktree while writing under a separate dirty checkout and require provenance to stay
bound to the clean executing source.

- [ ] **Step 6: Run Task 6 gates and commit**

```bash
uv run pytest tests/learning/test_artifacts.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/learning/artifacts.py tests/learning/test_artifacts.py
git commit -m "feat: persist learned-policy artifacts"
```

---

### Task 7: Compact Oracle Dataset and Behavior-Cloning Artifact

**Files:**
- Create: `src/harpy/learning/bc.py`
- Create: `tests/learning/test_bc_dataset.py`
- Create: `tests/learning/test_bc_training.py`

**Interfaces:**
- Consumes Task 1 BC splits/profiles, Task 3 cached evidence, Task 4 observation and
  BC-network contracts, Task 5 evaluator/criterion, and Task 6 artifact writer.
- Produces compact `BCExample` records, `OracleTrajectoryDataset`, class weights,
  deterministic BC training, held-out accuracy, model save/load, and
  `train_bc_artifact`.
- Hidden cents truth may produce oracle labels and index the evidence provider solely
  to reconstruct the exact actor-visible spectrum. It never enters scalar features or
  a serialized actor input.

- [ ] **Step 1: Write compact-dataset RED tests**

Generate examples by stepping a real cached environment with the exact
`minimum_action_plan`:

```python
def test_examples_are_real_minimum_plan_trajectories() -> None:
    episode = EpisodeSpec(target_note_index=12, source_pitch_cents=5_151)
    examples = build_oracle_examples((episode,), env_factory=cached_env_factory)
    assert examples[-1].action is PitchAction.SUBMIT
    assert tuple(example.action for example in examples) == minimum_action_plan(
        5_151 - 6_000,
        ControlState(),
        tolerance_cents=5,
    )
    assert all(not hasattr(example, "spectrum") for example in examples)
    assert all(not hasattr(example, "error_cents") for example in examples)
```

Pin one example per real state before its labeled action, decreasing
`steps_remaining`, Submit last, no post-terminal example, immutable records, source
episode order, env close on success/failure, and exact rejection of a planner/env
trajectory mismatch. The example schema is:

```python
@dataclass(frozen=True, slots=True)
class BCExample:
    episode: EpisodeSpec
    controls: ControlState
    steps_remaining: int
    action: PitchAction
```

- [ ] **Step 2: Run dataset tests and observe RED**

```bash
uv run pytest tests/learning/test_bc_dataset.py -q
```

Expected: missing `harpy.learning.bc`.

- [ ] **Step 3: Implement compact examples and the lazy dataset**

Use:

```python
def build_oracle_examples(
    episodes: Sequence[EpisodeSpec],
    *,
    env_factory: Callable[[], gymnasium.Env],
) -> tuple[BCExample, ...]: ...


class OracleTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Sequence[BCExample],
        evidence_provider: SpectrumEvidenceProvider,
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[PolicyObservation, np.int64]: ...


def training_class_weights(actions: Sequence[PitchAction]) -> torch.Tensor: ...
```

For each item, compute effective cents as
`episode.source_pitch_cents + controls.offset_cents`, resolve evidence through the
cache, reconstruct the exact raw Gym fields, and call `preprocess_observation`.
Returned arrays are fresh and the label is a scalar `np.int64`. Dataset construction
must not pre-materialize spectra.

Require all seven labels and compute each training-only class weight as
`N / (7 * class_count)`. Reject empty examples, bool indices, missing classes,
malformed provider output, and nonfinite weights. Tests poison validation labels and
prove they never enter weight construction.

- [ ] **Step 4: Write deterministic-training RED tests**

Use Task 1's dependency-free `BCEpochMetrics`/`BCTrainingSummary` records and pin the
training functions:

```python
def train_behavior_cloning(
    training_dataset: OracleTrajectoryDataset,
    validation_dataset: OracleTrajectoryDataset,
    *,
    config: BCProfile,
    seed: int,
    device: torch.device,
) -> tuple[BCPolicyNetwork, BCTrainingSummary]: ...


@torch.inference_mode()
def next_action_accuracy(
    model: BCPolicyNetwork,
    dataset: OracleTrajectoryDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float: ...
```

Script validation losses to prove: only a strictly lower loss selects a checkpoint;
ties retain the earlier epoch and do not reset patience; checkpoint patience of five
counts consecutive non-improvements; smoke runs both epochs because patience is
disabled; and selected weights, not final weights, are returned. Assert seeded
shuffle, `num_workers=0`, ordered validation, AdamW, checked-in learning rate/weight
decay, training-derived weights for both CE calculations, finite metrics, and two
identical CPU runs producing identical histories/state dicts apart from wall time.

Aggregate weighted CE at dataset level: request unreduced losses, sum
`class_weight[label] * negative_log_likelihood` over examples, then divide by
`sum(class_weight[label])`. Accuracy is total correct divided by total examples.
Tests use unequal final batches and two validation batch partitions and require the
same validation loss, selected epoch, and accuracy.

- [ ] **Step 5: Implement BC training, safe persistence, and actor reload**

Seed Python, NumPy, Torch, the DataLoader generator, and deterministic Torch mode
before constructing the model or optimizer. Clone the selected state dict onto CPU.
Implement:

```python
def save_bc_model(path: Path, model: BCPolicyNetwork) -> None: ...


def validate_bc_artifact(
    artifact: LoadedArtifact | PendingArtifactView,
) -> None: ...


def load_bc_actor(
    artifact: LoadedArtifact | PendingArtifactView,
    *,
    device: DeviceName = DeviceName.CPU,
) -> BCActor: ...
```

Save `state_dict` only. Load with the pinned Torch release's safest weights-only mode
into a code-constructed architecture. Validate exact parameter names, shapes, dtypes,
parameter count, architecture ID, preprocessing ID, and finite values before exposing
the actor; never partially load.
`validate_bc_artifact` first applies Task 6's strict config/summary codecs, then
cross-checks trainer/profile/seed, split and suite digests, counts, schema IDs, and
manifest inventory. A writer pending view validates exactly the names declared in that
view; when those names include evaluation files it decodes them with Task 5's
`EvaluationFile.from_document`. A complete artifact always requires and decodes every
evaluation file. `load_bc_actor` calls validation before
model construction; Task 9 calls it across all inputs before constructing any actor.
When passed a pending view, require the exact published config/summary/model names and
an incomplete manifest; when passed a loaded artifact, require a complete manifest.
No arbitrary path or caller-constructed manifest overload is accepted.

- [ ] **Step 6: Write the BC workflow RED tests**

With fakes around training, persistence, loading, and evaluation, assert this exact
order:

1. validate profile/seed/device/output and capture source status;
2. publish the incomplete manifest;
3. build deterministic training/validation examples and train;
4. publish config, summary, and `model.pt`;
5. request a writer `pending_view` for config/summary/model and reload its persisted
   `model.pt` through `load_bc_actor` on CPU;
6. for checkpoint only, compute held-out IID next-action accuracy exactly once after
   selection; smoke may compute a smoke-suite diagnostic but never touches final IID;
7. run the reloaded actor closed-loop on only the profile suite(s), including probes;
8. compute the BC criterion through `evaluate_bc_criterion` only;
9. publish evaluation JSON, request a full-inventory pending view, and strictly
   validate every config/summary/model/evaluation payload;
10. derive/publish the complete manifest last and return a fresh `load_artifact`
    result.

The train command's internal persisted-model evaluation is CPU even after explicit
exploratory CUDA training, matching the public load default; both training and
evaluation device are recorded separately.

Inject a failure at each boundary and require the post-bootstrap manifest to stay
incomplete. Poison final IID outcomes and prove they cannot affect epoch selection,
configuration, architecture, or weights. A smoke artifact is always ineligible. A BC
checkpoint is eligible only for seed 0, CPU, and clean committed source; another seed
or CUDA remains a valid exploratory artifact.

- [ ] **Step 7: Implement `train_bc_artifact` without a generic trainer layer**

```python
def train_bc_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> LoadedArtifact: ...
```

Call `build_bc_episode_splits(profile=profile, run_seed=seed)` for every run. Verify
Task 1's golden split digests only for declared seed 0; record the actual deterministic
digests for every exploratory seed, and test a repeatable nonzero seed is distinct and
ineligible. Use Task 6's atomic writer. For smoke publish
`evaluation-smoke.json`; for checkpoint publish `evaluation-iid.json` and
`evaluation-ood.json`. Store wall time as descriptive metadata, but exclude it from
deterministic-equality claims. Complete with `BCTrainingCounts` containing profile
episode counts plus actual trajectory-example counts, cross-checked against summary
and dataset lengths. Do not write a model until training has completed and never load
BC weights into any PPO path.

- [ ] **Step 8: Run Task 7 gates and commit**

```bash
uv sync --locked --group train
uv run pytest tests/learning/test_bc_dataset.py tests/learning/test_bc_training.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/learning/bc.py tests/learning/test_bc_dataset.py \
  tests/learning/test_bc_training.py
git commit -m "feat: train behavior-cloned sine policy"
```

---

### Task 8: Fresh PPO Training and Budget-Terminal Semantics

**Files:**
- Create: `src/harpy/learning/ppo.py`
- Create: `tests/learning/test_ppo.py`

**Interfaces:**
- Consumes Tasks 1, 3, 4, 5, and 6. It does not import Task 7's BC module.
- Produces the Harpy budget latch/VecEnv adapter, exact SB3 model construction,
  deterministic PPO training, PPO save/load, and `train_ppo_artifact`.
- The public Gym five-tuple remains unchanged; only SB3's private timeout-bootstrap
  interpretation is corrected.

- [ ] **Step 1: Write raw-latch and VecEnv RED tests**

Define the intended surface:

```python
class BudgetExhaustionLatch(gymnasium.Wrapper):
    def consume_budget_exhausted(self) -> bool: ...


class NoBudgetBootstrapVecEnv(VecEnvWrapper):
    def step_wait(self) -> VecEnvStepReturn: ...


def make_ppo_vec_env(
    env_factory: Callable[[], gymnasium.Env],
) -> NoBudgetBootstrapVecEnv: ...
```

Force 64 non-Submit actions through the real environment. Assert the latch validates
the terminal `EpisodeResult` before `DummyVecEnv` auto-reset and tags only a genuine
`TerminalReason.BUDGET_EXHAUSTED`. The raw observation, reward, `terminated=False`,
`truncated=True`, actor-safe info, and result remain identical. Submit termination is
unchanged. A normal non-Harpy `TimeLimit` truncation is not tagged.

- [ ] **Step 2: Run timeout tests and observe RED**

```bash
uv run pytest tests/learning/test_ppo.py -q -k "budget or bootstrap or timeout"
```

Expected: missing `harpy.learning.ppo`.

- [ ] **Step 3: Implement the validated bootstrap adapter**

Build the nesting as
`DummyVecEnv([lambda: PolicyObservationWrapper(BudgetExhaustionLatch(env))])` and keep
the latch reference in `NoBudgetBootstrapVecEnv`. The latch records a one-shot private
boolean and `consume_budget_exhausted()` clears it; the marker never enters Gym info.
`NoBudgetBootstrapVecEnv.step_wait` consumes that latch after the inner VecEnv returns
and changes only SB3's copied `TimeLimit.truncated` interpretation for the validated
Harpy budget case, so SB3 does not add `gamma * V(terminal_observation)`.

Collect one real PPO rollout with a mocked nonzero terminal value and assert the final
rollout-buffer reward is bit/equality-compatible with Harpy's exact returned reward.
Mutation-test removing the adapter: the same assertion must fail because the terminal
value is added. Separately prove ordinary time limits still bootstrap.

- [ ] **Step 4: Write exact PPO-construction RED tests**

Use Task 1's `PPOTrainingSummary` and pin:

```python
def make_ppo_model(
    env: VecEnv,
    *,
    config: PPOProfile,
    seed: int,
    device: str,
) -> PPO: ...


def train_ppo(
    env: VecEnv,
    *,
    config: PPOProfile,
    seed: int,
    device: str = "cpu",
) -> tuple[PPO, PPOTrainingSummary]: ...
```

Spy on the SB3 constructor and require exactly:

```python
PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": HarpySineFeaturesExtractor,
        "net_arch": [],
        "share_features_extractor": True,
        "normalize_images": False,
    },
    gamma=config.gamma,
    n_steps=config.n_steps,
    batch_size=config.batch_size,
    n_epochs=config.n_epochs,
    learning_rate=config.learning_rate,
    gae_lambda=config.gae_lambda,
    clip_range=config.clip_range,
    ent_coef=config.ent_coef,
    vf_coef=config.vf_coef,
    seed=seed,
    device=device,
)
```

Assert one synchronous env, no `VecNormalize`, no observation/reward normalization,
no masks, no evaluator-truth callback, direct shared action/value heads, and exactly
75,816 policy parameters. Both profiles' total timesteps are rollout aligned and the
observed completed count must equal the requested count.

- [ ] **Step 5: Implement deterministic fresh PPO training**

Seed Python, NumPy, Torch, SB3, Gym, and the scheduled training stream before model
construction. `make_ppo_model` has no model/checkpoint/BC/load parameter. Add a test
that poisons every Task 7 loader and still trains PPO from its fresh seeded state.
Use deterministic inference only for evaluation, not stochastic training action
selection.

Implement:

```python
def save_ppo_model(path: Path, model: PPO) -> None: ...


def validate_ppo_artifact(
    artifact: LoadedArtifact | PendingArtifactView,
) -> None: ...


def load_ppo_actor(
    artifact: LoadedArtifact | PendingArtifactView,
    *,
    device: DeviceName = DeviceName.CPU,
) -> PPOActor: ...


def train_ppo_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> LoadedArtifact: ...
```

Validate the trusted-local archive, manifest contract IDs, policy topology, parameter
count, and requested device. Evaluation defaults to CPU even for a CUDA-trained
exploratory artifact. Do not claim SB3 `.zip` is safe for untrusted input.
`validate_ppo_artifact` applies the same pending-versus-complete strict codec and
cross-field rules before model construction; complete PPO evaluation files must have
`next_action_accuracy=None`.
Apply the same pending-versus-complete status and exact-name rules as the BC loader;
never accept a loose model path.

- [ ] **Step 6: Write and satisfy PPO artifact-order RED tests**

Use the same atomic order as BC: incomplete manifest → fresh train → config/summary/
`model.zip` → writer pending view → reload persisted PPO actor on CPU → profile evaluation/probes → evaluation JSON
→ full-inventory pending validation → complete manifest → strict reload. Inject failure at every boundary. Assert a PPO
artifact never stores the aggregate five-seed criterion; only the later compatible
multi-artifact report can own it. Smoke, CUDA, dirty source, and undeclared seeds are
complete but ineligible. Complete with `PPOTrainingCounts` whose requested and
completed steps both equal the rollout-aligned profile budget.

- [ ] **Step 7: Run Task 8 gates and commit**

```bash
uv sync --locked --group train
uv run pytest tests/learning/test_ppo.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/learning/ppo.py tests/learning/test_ppo.py
git commit -m "feat: train PPO sine policy"
```

---

### Task 9: Cross-Trainer Workflows, Trace, and `harpy-sine-learn` CLI

**Files:**
- Create: `src/harpy/learning/workflows.py`
- Create: `src/harpy/learning/trace.py`
- Create: `src/harpy/learning/cli.py`
- Create: `tests/learning/test_workflows.py`
- Create: `tests/learning/test_trace.py`
- Create: `tests/learning/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Integrates Tasks 5–8 without introducing a registry, database, generic experiment
  platform, or duplicate metric/gate implementation.
- Produces strict multi-artifact evaluation, complete hands-on traces, four argparse
  commands, and the console entry point.
- CLI parsing/help remains Torch/SB3/Qt-free; heavy imports occur only after command
  validation and dependency checks.

- [ ] **Step 1: Write multi-artifact workflow RED tests**

Pin:

```python
def evaluate_artifacts(
    artifact_paths: Sequence[Path],
    *,
    device: DeviceName = DeviceName.CPU,
) -> EvaluationReport: ...


def run_artifact(
    artifact_path: Path,
    *,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> EpisodeTrace: ...
```

Resolve aliases before duplicate detection. Validate every complete artifact and all
compatibility fields before constructing any actor. Reject duplicate paths,
duplicate `(trainer, seed)` rows, unknown/extra files, mixed profiles, and mismatched
environment/suite/grid/preprocessing/policy-architecture versions. The PPO-only value
head is not a BC/PPO policy-architecture mismatch.

Canonical output is BC seed order, then PPO seed order, independent of argv. Add
Random, Reward Search, Spectrum Peak, and Oracle exactly once. Call Task 5 aggregation
and criteria APIs; never recompute formulas. Compute the PPO criterion only for the
exact eligible CPU set `{0,1,2,3,4}` with no extra PPO seed; accept at most eligible
BC seed 0 in that criterion-bearing checkpoint report. Exploratory reports still
serialize rows but mark criteria ineligible.

Derive eligibility only from the strict manifests: checkpoint profile, declared seed,
CPU training/evaluation, clean source, committed required inputs, compatible contract
IDs, and complete validated inventory. End-to-end tests mutate each provenance field
(profile, device, dirty flag, required-input flag, seed) and prove it cannot reach a
criterion API as `eligible=True`.

- [ ] **Step 2: Implement explicit workflows**

`workflows.py` may dispatch between the two explicit trainer loaders, but it must not
create a plug-in registry or a generic serializer. Validate the entire input set before
calling `load_bc_actor` or `load_ppo_actor`. The resulting `EvaluationReport` owns
ordered rows, BC/PPO criteria, compatible contract IDs, and strict JSON conversion.

```python
@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: int
    profile: ProfileName
    evaluation_device: DeviceName
    environment_contract_id: str
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    rows: tuple[EvaluationRow, ...]
    bc_criterion: BCScientificCriterion | None
    ppo_criterion: PPOScientificCriterion | None

    def to_document(self) -> dict[str, JSONValue]: ...
    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> EvaluationReport: ...


def evaluation_report_bytes(report: EvaluationReport) -> bytes: ...
def evaluation_report_from_bytes(content: bytes) -> EvaluationReport: ...
```

The codec rejects nonfinite metrics and writes paired terminal records needed for the
declared comparisons. It includes the artifact's already-recorded descriptive training
wall time, but excludes fresh evaluation timing, invocation timestamps, and input
paths from deterministic report bytes. `evaluation_report_from_bytes` calls Task 6's
public `decode_json_bytes` and then the strict `from_document` codec; it does not add a
second permissive JSON parser.

Public `evaluate` and `run` reject incomplete artifacts. `evaluate` never mutates an
artifact; its optional report path is outside all input artifact roots and uses
`write_new_bytes`.

- [ ] **Step 3: Write complete-trace RED tests**

Define:

```python
@dataclass(frozen=True, slots=True)
class TraceStep:
    step: int
    action: PitchAction
    reward: float


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    environment_id: str
    distribution_id: str
    seed: int
    target_note_index: int
    target_note: str
    steps: tuple[TraceStep, ...]
    terminal_reason: TerminalReason
    final_absolute_error_cents: int
    submitted_success: bool
    action_count: int
    excess_actions: int | None
    total_return: float


def trace_episode(actor: Actor, *, seed: int) -> EpisodeTrace: ...
def format_human_trace(trace: EpisodeTrace) -> str: ...
def trace_json_bytes(trace: EpisodeTrace) -> bytes: ...
```

Use the direct frozen `Harpy/SinePitch-v0.reset(seed=N)` full-range distribution, not
the central training distribution or a fixed evaluation suite. Poison `episode_result`
before done and prove the actor receives only the observation. Human output prints the
target first, every action/reward next, and terminal truth last. JSON is one strict
finite sorted object plus one newline; no absolute paths, timestamps, source pitch, or
preterminal error appears. Same actor/artifact/seed repeats byte-for-byte.

- [ ] **Step 4: Implement trace capture and rendering**

Map target index to a human note label through existing tuning helpers. Validate every
learned action before `env.step`. Close the environment under all exits. Read and
validate `EpisodeResult` only after terminal/truncated. Render `excess_actions` as
JSON `null` when undefined and label the distribution explicitly as full-range
demonstration evidence.

- [ ] **Step 5: Write CLI and exit-code RED tests**

Pin the exact grammar:

```text
train-bc  --profile {smoke,checkpoint} --seed N --output PATH [--device {cpu,cuda}]
train-ppo --profile {smoke,checkpoint} --seed N --output PATH [--device {cpu,cuda}]
evaluate  ARTIFACT [ARTIFACT ...] [--output FILE] [--device {cpu,cuda}]
run       ARTIFACT --seed N [--json] [--device {cpu,cuda}]
```

Tests require argument/closed-contract errors `2`, dependency/artifact/I/O/training/
evaluation failures `1`, handled `KeyboardInterrupt` `130`, and success—including
`criterion_not_met`—`0`. Reject bool-like, negative, and noninteger seeds before a
workflow call. Failure stdout is empty and stderr is one concise diagnostic without a
traceback.

Pin the case-level mapping rather than relying on whichever exception happens to leak:

| Case | Exit |
| --- | ---: |
| Existing artifact/report output, empty or non-directory input, equal/nested report path, duplicate artifact/seed, compatibility mismatch, unwritable output ancestor | 2 |
| Malformed/incomplete/hash-bad artifact, missing training dependency, unavailable requested CUDA, valid-path publication I/O failure, trainer/evaluator failure | 1 |
| Handled user interrupt | 130 |

For existing, nested, or unwritable `evaluate --output`, resolve and preflight the path
before calling `evaluate_artifacts` or constructing any actor. `write_new_bytes`
remains the final race-safe no-clobber publication. Tests poison workflow/actor calls
and prove every path-contract failure happens first.

Assert `evaluate` stdout bytes equal `--output` bytes; output is create-only and
cannot equal or sit under an artifact root. `run --json` emits only JSON; human mode
is readable and complete. Emit the trusted-local model warning once for public
evaluate/run. Help never imports Torch/SB3/Qt.

- [ ] **Step 6: Implement argparse and the entry point**

```python
def main(argv: Sequence[str] | None = None) -> int: ...


if __name__ == "__main__":
    raise SystemExit(main())
```

Append exactly this row to the existing `[project.scripts]` table:

```toml
harpy-sine-learn = "harpy.learning.cli:main"
```

Preserve the existing `harpy` and `harpy-sine-gym` scripts. Create missing output
parents recursively only after checking the nearest existing ancestor is a writable
directory; the final artifact/report target remains exclusive. An explicit CUDA
request fails if unavailable rather than falling back.

- [ ] **Step 7: Run Task 9 gates and commit**

```bash
uv lock --check
uv run pytest tests/learning/test_workflows.py tests/learning/test_trace.py \
  tests/learning/test_cli.py -q
uv run --isolated --no-group train --locked harpy-sine-learn --help
uv run --isolated --no-group train --locked python -m harpy.learning.cli --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add pyproject.toml uv.lock src/harpy/learning/workflows.py \
  src/harpy/learning/trace.py src/harpy/learning/cli.py \
  tests/learning/test_workflows.py tests/learning/test_trace.py \
  tests/learning/test_cli.py
git commit -m "feat: add learned sine policy CLI"
```

---

### Task 10: Real Optional-Stack Smoke and Hands-On Documentation

**Files:**
- Create: `tests/learning/test_training_smoke.py`
- Modify: `README.md`
- Create: `docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md`

**Interfaces:**
- Exercises the actual Torch/SB3 stack and all four CLI commands on the bounded smoke
  profile.
- Documents commands a user can run immediately and creates the acceptance template
  required by Task 11.
- Does not modify `docs/project-notebook.md` because the root checkout has the user's
  unrelated edit.

- [ ] **Step 1: Write real smoke RED tests**

In `test_training_smoke.py`, run the smallest checked-in smoke configuration on CPU
through production workflows. Assert BC and PPO each:

- BC uses the exact configured training/validation episode counts and records its
  actual trajectory-derived train/validation example counts; PPO completes the exact
  configured environment timesteps;
- publish the exact closed inventory and hashes;
- reload the persisted model rather than reuse the in-memory trainer object;
- evaluate and complete atomically;
- produce deterministic reloaded actions for a fixed observation;
- run one full-range episode through the trace path;
- emit finite strict JSON with distinct learned/baseline rows.

The test module may use the `train`-group availability marker. It skips only when the
optional group is genuinely absent. A separate no-group subprocess test must print
`uv sync --group train`, create no output, and never import Qt.

That no-group test must use a genuinely fresh environment, not the shared `.venv`:
launch exactly
`uv run --isolated --no-group train --locked harpy-sine-learn train-bc --profile smoke --seed 0 --output <tmp/new> --device cpu`
with a temp nonexistent output and a temporary `sitecustomize.py` whose import finder
raises `ModuleNotFoundError` for `torch`, `stable_baselines3`, and `PySide6`. Assert
exit 1, exact install guidance, empty stdout, and that the output path was never
created.

- [ ] **Step 2: Run the real smoke characterization and fix only proven gaps**

```bash
uv sync --locked --group train
uv run pytest tests/learning/test_training_smoke.py -q -rs
```

Because Tasks 7–9 already own the production behavior, this acceptance test may be
GREEN on its first run. Record that honestly. If it exposes a defect, reproduce the
defect in the owning module's focused test, observe RED, apply the smallest fix, run
that module's full focused slice plus this smoke file, and commit the scoped production
and regression fix before editing documentation. Do not reduce profile sizes, bypass
persistence, or fake SB3/Torch to turn this green.

- [ ] **Step 3: Write README and acceptance template**

README must include:

- `uv sync --group train` as the only learned-policy install step;
- the four smoke commands exactly as specified;
- CPU default and explicit exploratory CUDA behavior;
- trusted-local artifact warning;
- direct explanation that BC is a diagnostic, PPO is fresh, Spectrum Peak is a
  control, and candidate spectrum is already visible in `SinePitch-v0`;
- engineering-vs-scientific outcome language;
- no claim that tools, recorded-audio pitch shifting, chords, or waveform
  generalization exist yet.

Create the acceptance document with exact headings for commit/lock/runtime, full
automated gates, BC smoke, PPO smoke, checkpoint artifacts, IID/OOD rows, lower/upper
OOD subsets, four baselines, zero/shuffled probes, BC/PPO criteria, payload hashes,
and honest concerns/pending items. At this point checkpoint fields remain explicitly
`pending`; do not invent results.

- [ ] **Step 4: Commit the documentation/template before acceptance runs**

```bash
git diff --check
git add README.md tests/learning/test_training_smoke.py \
  docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md
git commit -m "docs: add learned sine policy workflow"
git status --short
```

The implementation worktree must be clean after this commit. If it is not, stop before
training; do not mark a dirty run eligible.

- [ ] **Step 5: Run fresh engineering gates and real smoke commands**

Use new paths; never delete or reuse an old run directory:

```bash
set -euo pipefail
uv sync --locked --group train
uv run pytest tests/learning/test_training_smoke.py -q -rs \
  | tee /tmp/harpy-milestone-d-training-smoke.txt
if rg -q "skipped" /tmp/harpy-milestone-d-training-smoke.txt; then exit 1; fi
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check

uv run harpy-sine-learn train-bc \
  --profile smoke --seed 0 --output runs/milestone-d-bc-smoke
uv run harpy-sine-learn train-ppo \
  --profile smoke --seed 0 --output runs/milestone-d-ppo-smoke
uv run harpy-sine-learn evaluate \
  runs/milestone-d-bc-smoke runs/milestone-d-ppo-smoke \
  --output runs/milestone-d-smoke-report.json
uv run harpy-sine-learn run runs/milestone-d-ppo-smoke --seed 123
uv run harpy-sine-learn run runs/milestone-d-ppo-smoke --seed 123 --json \
  > runs/milestone-d-run-1.json
uv run harpy-sine-learn run runs/milestone-d-ppo-smoke --seed 123 --json \
  > runs/milestone-d-run-2.json
cmp runs/milestone-d-run-1.json runs/milestone-d-run-2.json
```

Record exact exits, elapsed times, artifact/report hashes, rows, and any warning. Smoke
criteria remain ineligible by design. Update the acceptance document with these real
engineering results and commit it before the scientific checkpoint:

```bash
git add docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md
git commit -m "docs: record milestone D smoke acceptance"
git status --short
```

---

### Task 11: Clean Five-Seed Checkpoint, Whole-Branch Review, and Evidence

**Files:**
- Modify: `docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md`
- Modify only when a validated review finding requires it: scoped production/test
  files from Tasks 1–10.

**Interfaces:**
- Runs the accepted BC seed 0 and PPO seeds 0–4 from one clean committed CPU source,
  produces the canonical comparison report, records weak or strong outcomes honestly,
  and closes Milestone D only after independent review and fresh verification.
- Checkpoint artifacts live outside a disposable linked worktree and stay ignored;
  source code and docs, not binary model files, are committed.

- [ ] **Step 1: Prove the checkpoint source is clean and authoritative**

From the implementation worktree:

```bash
git status --short
git rev-parse HEAD
sha256sum uv.lock
uv sync --locked --group train
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check 957c140...HEAD
git status --short
```

Require zero status output both before and after sync/tests and immediately before the
first trainer command. Record Python, NumPy, Gymnasium, Torch, SB3, platform,
CPU, thread settings, and available CUDA details. CPU remains selected. Do not start if
any required config/suite/lock input is uncommitted.

- [ ] **Step 2: Run one BC and exactly five PPO checkpoint artifacts**

Choose one new, explicit root outside any removable linked worktree, for example
`/home/haydenw/Projects/Harpy/runs/milestone-d-v1`, and require that it does not exist.
Create its parent through the CLI; never `rm`, overwrite, or resume it.

```bash
set -euo pipefail
verify_harpy_artifact() {
  uv run python -c \
    'from pathlib import Path; import sys; from harpy.learning.artifacts import load_artifact; load_artifact(Path(sys.argv[1]))' \
    "$1"
}
uv run harpy-sine-learn train-bc --profile checkpoint --seed 0 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1/bc-0
verify_harpy_artifact /home/haydenw/Projects/Harpy/runs/milestone-d-v1/bc-0
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 0 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-0
verify_harpy_artifact /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-0
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 1 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-1
verify_harpy_artifact /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-1
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 2 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-2
verify_harpy_artifact /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-2
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 3 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-3
verify_harpy_artifact /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-3
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 4 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-4
verify_harpy_artifact /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-4
```

Run sequentially by default to keep the CPU comparison stable and avoid six trainers
contending for memory/threads. After each command, load/verify the complete artifact
and record exit/time before beginning the next. A criterion miss is not a rerun
trigger. A genuine execution failure leaves an incomplete artifact; preserve it and
use a newly versioned output root after fixing the defect.

After each trainer command, run the strict loader against that one directory. After all
six, verify shared provenance and every payload—not merely `manifest.json` hashes:

```bash
uv run python - <<'PY'
from pathlib import Path
import hashlib
import subprocess
from harpy.learning.artifacts import load_artifact

root = Path("/home/haydenw/Projects/Harpy/runs/milestone-d-v1")
artifacts = [load_artifact(root / "bc-0")]
artifacts.extend(load_artifact(root / f"ppo-{seed}") for seed in range(5))
source = artifacts[0].manifest.source
assert all(item.manifest.source == source for item in artifacts)
assert not source.dirty_tree and source.required_inputs_committed
assert source.commit == subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
assert source.dependency_lock_sha256 == hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
assert all(item.manifest.status.value == "complete" for item in artifacts)
assert all(item.manifest.runtime.device.value == "cpu" for item in artifacts)
assert all(item.manifest.evaluation_device.value == "cpu" for item in artifacts)
PY
```

- [ ] **Step 3: Produce the canonical aggregate and hands-on trace**

```bash
set -euo pipefail
uv run harpy-sine-learn evaluate \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/bc-0 \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-0 \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-1 \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-2 \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-3 \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-4 \
  --output /home/haydenw/Projects/Harpy/runs/milestone-d-v1-report.json
uv run harpy-sine-learn run \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-0 --seed 123
sha256sum \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/*/manifest.json \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1-report.json
uv run python - <<'PY'
from pathlib import Path
from harpy.learning.workflows import evaluation_report_bytes, evaluation_report_from_bytes

path = Path("/home/haydenw/Projects/Harpy/runs/milestone-d-v1-report.json")
content = path.read_bytes()
assert evaluation_report_bytes(evaluation_report_from_bytes(content)) == content
PY
```

The strict artifact loop above is the authoritative inventory/payload hash
verification; `sha256sum` is retained only as a compact evidence index.

The report must contain BC, five PPO seeds, Random, Reward Search, Spectrum Peak, and
Oracle; IID and OOD combined plus lower/upper OOD rows; zero/shuffled probes; exact
paired terminal records; and separate BC/PPO criteria. Do not suppress a weak seed,
tune after seeing final suites, or translate `criterion_not_met` into an execution
failure.

- [ ] **Step 4: Request independent whole-branch reviews**

Dispatch at least three read-only reviewers in parallel:

1. environment/cache/leakage and frozen Milestone C semantics;
2. network/BC/PPO numerical and scientific protocol;
3. artifacts/CLI/error/atomicity and documentation claims.

Each reviewer must inspect `957c140...HEAD`, the approved spec, real checkpoint
report, and relevant tests; classify Critical/Important/Minor findings with file/line
evidence. The primary agent reconciles all findings. For every accepted defect, use
`superpowers:receiving-code-review`, write a focused failing regression, observe RED,
apply the smallest fix, rerun the focused slice, and commit a scoped fix. Do not patch
model outcomes or rerun merely to improve a criterion.

- [ ] **Step 5: Run final fresh verification**

After all fixes and with no training process active:

```bash
set -euo pipefail
uv sync --locked --group train
uv run pytest tests/learning/test_training_smoke.py -q -rs \
  | tee /tmp/harpy-milestone-d-final-training-smoke.txt
if rg -q "skipped" /tmp/harpy-milestone-d-final-training-smoke.txt; then exit 1; fi
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import sys, harpy.learning; assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules"
uv run --isolated --no-group train --locked harpy-sine-learn --help
git diff --check
git diff --check 957c140...HEAD
git status --short
set -o noclobber
uv run harpy-sine-learn run \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-0 --seed 123 --json \
  > /home/haydenw/Projects/Harpy/runs/milestone-d-v1-trace-1.json
uv run harpy-sine-learn run \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1/ppo-0 --seed 123 --json \
  > /home/haydenw/Projects/Harpy/runs/milestone-d-v1-trace-2.json
cmp /home/haydenw/Projects/Harpy/runs/milestone-d-v1-trace-1.json \
  /home/haydenw/Projects/Harpy/runs/milestone-d-v1-trace-2.json
```

Verify every artifact hash and the aggregate report without retraining. If a code fix changes an artifact
contract or actor behavior, mark the previous scientific run invalid and repeat the
entire checkpoint under a new create-only root; never silently reuse it. Apply this
invalidation matrix explicitly:

- docs/test-only changes that cannot affect runtime evidence: no evidence rerun;
- report presentation/ordering only, with identical validated rows and criteria:
  regenerate the report at a new create-only path;
- any suite, training stream/config, environment/cache, preprocessing, network/actor,
  trainer, internal evaluation rollout, metric/probe/criterion, serialization, or
  artifact-content change: rerun all six training artifacts and the report under a new
  root.

If invalidation creates `milestone-d-v2` or later, replace every Step 5 trace/report
path with that final accepted root; never fall back to the stale v1 evidence.

- [ ] **Step 6: Complete and commit the acceptance record**

Fill the acceptance document with the exact clean commit and lock hash; all commands
and exits; dependency/device data; training times/examples/steps; every learned and
baseline row; diagnostics; BC and PPO criterion objects; artifact inventories and
hashes; reviewer verdicts; and honest concerns. State `engineering_passed` separately
from `criterion_met` for each model.

```bash
git add docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md
git commit -m "docs: record milestone D learned policy checkpoint"
git status --short
```

The root checkout's unrelated `docs/project-notebook.md` edit remains untouched and
uncommitted throughout. Then use `superpowers:finishing-a-development-branch` to offer
review/push/PR or local merge options; do not publish without the user's authorization.
