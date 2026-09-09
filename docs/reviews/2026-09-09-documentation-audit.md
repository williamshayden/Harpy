# Documentation verification — 9 September 2026

The current revised-v1 documentation was checked against the implementation,
installed packages, and retained scientific evidence. This was a documentation
audit: production code, tests, research sources, frozen results, and checkpoint
weights are unchanged. The revision remains Unreleased; nothing was published.

## Corrections

- Pitch selects checkpoints using clean validation. PPO saves its final training
  step without validation selection or a performance gate. The specification and
  practical guide now state that distinction.
- The waveform Python example runs a clean episode in memory. The adjacent CLI
  example saves a noisy episode; they are no longer described as equivalent.
- The committed controller uses the target and controls; the environment enforces
  the action budget. Zero estimated planning tolerance is distinguished from
  five-cent class spacing and the five-cent true-error success bound.
- The guide identifies later trace estimates as `None`, meaning no new inference.
- Confirmation membership is already evaluated release evidence. A new final
  confirmation claim needs fresh frozen membership. Complete historical results
  and seed-1/2 artifacts require separate local archives, beyond the source tree.
- Migration names the actual old `train-pitch` and `train-ppo` commands, and limits
  the historical weight-import description to the implemented pitch conversion.
- Acceptance separates dated scientific qualification from subsequent engineering
  checks. Its linked release-review document is now included in the sdist.
- Two historical reference targets were repaired: the versioned
  [librosa API page](https://librosa.org/doc/0.11.0/generated/librosa.effects.pitch_shift.html)
  and the [author-hosted NIME paper](https://github.com/vincenzomadaghiele/RL-synth-control/blob/main/paper/RLsynth_NIME26.pdf).
  The former DOI could not be verified; the replacement PDF location was verified
  independently. Historical claims and command bodies otherwise remain intact.

## Executed examples

All **23 current documented Harpy CLI examples**, all **three Python snippets**,
and the standalone custom-actor example were exercised using installed code
outside the checkout. The 15 fenced example blocks and standalone script were
compared with the final documentation; their executable contents are unchanged.

The base examples produced nine validated results with 64 successful actor-episodes.
Optional workflows installed Harpy through `pip install '.[pitch]'` and
`pip install '.[train]'`, then ran the documented reference, pitch training,
PPO training, and saved-model evaluation commands. Pitch and reference commands
ran with Stable-Baselines3 imports explicitly blocked.

The shipped reference and newly trained pitch model each succeeded on the six
smoke episodes. PPO training completed 2,048 environment steps and its evaluation
produced six valid failed submissions, with no budget truncations. Command
completion establishes the workflow, not useful PPO tuning performance; the guide
now explicitly explains this distinction. All saved outputs were strictly loaded
and read in text, Markdown, and CSV.

The optional environment installed a fresh Harpy package while sharing the
existing qualified Torch/SB3 dependencies. It was not a fresh download of the full
ML stack. An initial audit post-check used a nonexistent `terminal.truncated`
field; it was corrected to inspect `terminal_reason`. All documented commands had
already succeeded; the harness failure and corrected validation are both retained.

## Links, packages, and scientific claims

The final inventory covered 48 existing documentation files across the repository
and external publication handoff: **118 local links and five heading fragments**
resolve. Historical plans retain proposed or retired code paths as historical
context, not current instructions. Current examples contain no missing code paths.
External targets were checked separately; the two replacement URLs passed, and
the AIMC page was readable through the browser-backed web reader despite an
automated HTTP 403 response.

Fresh wheel and sdist environments passed **59 checks** on Linux/WSL with Python
3.12. Both base installations run without Torch/SB3, expose only the `harpy`
console entry point, and support the documented execution/readout workflows.
All **26 packaged documentation links** resolve; the included guide, README,
waveform notes, and release review match the source bytes.

Numerical claims were checked against the saved reference qualification and
representation study: **38,200 archived records** were reloaded with unchanged
canonical documents and hashes. No new checkpoint-profile training, full
650/1,000-pair qualification run, CUDA run, or native Windows/macOS qualification
was performed for this audit. Historical milestone commands were not executed
against the revised package; they remain tied to their documented historical tag.

## Evidence

Local evidence is under `outputs/docs-audit-20260909/`:

- `verification.json`, `documentation.patch`, and source hash snapshots.
- `current-examples/evidence/verification.json`: exact base examples and results.
- `optional-workflows/verification.json`: installation, reference, and real training commands.
- `claims-evidence.json`: implementation and archived numerical checks.
- `final-documentation-check.json`: final local links, anchors, and executable-content checks.
- `installed-verification/verification.json`: fresh wheel/sdist checks and hashes.
- `external-link-checks-final.json`: reference availability and repaired targets.

These checks establish the documented Linux/WSL Python 3.12 workflows at this
revision. They do not make the unpublished archives available to a fresh clone or
qualify arbitrary trained models.
