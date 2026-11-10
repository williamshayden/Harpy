# LLM sound-matching pilot

The next development priority is an LLM actor for the existing sound-matching
task. This is a research prototype in `research/sound_matching/llm/`, separate
from the frozen 2.0.0 study and the installed `harpy` CLI. Stable promotion stays
deferred until matching quality improves and new listening evidence supports it.
See the [listening diagnosis](v2-listening-review.md).

## What the model receives

HarnessBridge 0.1 provides text conversations over a local API. This pilot sends
declared features computed from the public target and current candidate audio,
the current editable patch, legal controls, budgets, and spectral distance. It
does not send raw audio, and it must not be described as native audio hearing.
It sends no target patch, target parameter labels, study manifest, file path,
audio hash, source-family identity, or evaluator-only pitch error.

The model returns one complete JSON patch proposal or an explicit submit:

```json
{"action":"submit"}
```

A proposal uses `{"action":"propose","patch":{...}}` with every required patch
field. Missing fields, duplicate JSON keys, nonfinite numbers, code fences, and
extra prose are rejected. The adapter does not silently repair a response or
make a second unrecorded model request. The ordinary task runner enforces edit
masks, legal values, and candidate-evaluation budgets.

The actor is a frozen general-purpose model controller. It is not RL training:
the pilot does not update the model's weights from rewards.

## Connection and authentication

Use an independently configured HarnessBridge instance with a supported native
Codex or Claude executable. The local bearer token authenticates the client to
the bridge; it is not a provider API key. Each harness owns its dedicated login.
`doctor` checks supported versions and sign-in status; a real completed request
is still needed to establish access to the configured model.

The current bridge accepts only text chat requests. It creates fresh isolated
harness processes and disables tools and repository access. Its configured alias
is not proof of a specific model's availability or successful inference. Record
the bridge/harness versions and alias mapping with any live pilot results.

For the existing Windows setup, complete the dedicated login when available:

```powershell
node "$env:USERPROFILE\Desktop\HarnessBridge\src\cli.js" login codex
node "$env:USERPROFILE\Desktop\HarnessBridge\src\cli.js" doctor codex
node "$env:USERPROFILE\Desktop\HarnessBridge\src\cli.js" serve
```

Keep the bridge bound to loopback. Do not copy desktop credentials into a
research artifact or put bearer tokens in command arguments, transcripts, or Git.

## Run from the source checkout

Use the qualified Python 3.12/Linux environment and a fresh output directory:

```bash
uv sync --locked --group dev
uv run --no-sync python -m research.sound_matching.llm.pilot --help
```

When Harpy and the bridge share the same loopback interface:

```bash
uv run --no-sync python -m research.sound_matching.llm.pilot \
  --base-url http://127.0.0.1:8790/v1 \
  --token-file /path/to/HarnessBridge/client.token \
  --model codex-default --suite tone-smoke --max-model-calls 4 \
  --output runs/llm-tone-pilot
```

For Harpy in WSL and HarnessBridge on Windows, use the provided
`bridge_client.mjs` through the command transport. Windows Node makes the
loopback request from Windows; no non-loopback listener or firewall change is
needed. `--command-json` names a local JSON file containing an argument array:
the Windows Node executable as visible to WSL, the relay's Windows-accessible
path, `--config`, and the Windows bridge configuration path. It must contain
paths and options only, never token values. Run:

```bash
uv run --no-sync python -m research.sound_matching.llm.pilot \
  --command-json /path/to/bridge-command.json \
  --model codex-default --suite tone-smoke --max-model-calls 4 \
  --output runs/llm-tone-pilot
```

Both connection modes require explicit model and transport choices. There is no
automatic hosted-provider fallback. The request and response sizes, client
deadline, and model-call count are bounded; no automatic request retry is used.

## Evidence and interpretation

The pilot compares the LLM actor with fixed, random-search, and pattern-search
actors on the same episodes through the ordinary durable runner. The usual
32-evaluation/64-decision ceilings apply. A separate per-episode LLM call cap is
also declared, and actual costs differ between actors. Deterministic controller
stops or restoration of the best heard patch are identified separately from a
model's own submit decision.

Requests, responses, parsing outcomes, reported usage, and latency are retained
in a separate journal. An attempt is recorded before contacting the transport.
Transport failure can leave the ordinary runner's model-call count unknown;
the request journal reports attempted calls separately. An HTTP attempt does
not establish how many provider-side inferences executed. Interrupted and failed
work must remain visible, not be relabeled as a successful model submission.

Immutable adapter sources and configuration are bound separately from the
growing request journals. Final manifests bind the saved evaluation and request
evidence. Recorded synthesis/actions can be replayed without calling a model.

Local fixture responses can qualify the plumbing, parsing, budgets, and evidence
workflow. They do not measure an LLM's matching ability. A live pilot needs actual
authenticated responses, matched comparisons, and target/match listening. Smoke
episodes demonstrate operation; quality improvement requires a broader
development comparison followed by newly frozen held-out confirmation. The old
final-test memberships cannot be reused to select the new actor.
