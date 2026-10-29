# WebArena data-gen pipeline (Agent SDK driver)

`scripts/webarena_datagen_pipeline.py` runs a whole task batch unattended: author → gate → repair →
verify → promote → export. It replaces the "Claude reads a TASK.md and drives `claude -p` by hand" loop
with a Python driver around the Claude Agent SDK, modelled on OSWorld-V2's `v15_datagen_pipeline.py`.

## Set up once

1. A small Brev (or any Ubuntu) instance is enough; nothing heavy runs on it. Clone this branch and have
   Claude Code follow [`SETUP.md`](../../SETUP.md) once: it checks out the `hub/` submodule
   (CUA-Gym-Hub) and serves the WebArena mocks on ports 8000–8004.
2. `pip install --user claude-agent-sdk==0.2.152` and log the `claude` CLI in (or use a gateway, below).
3. Map tasks only: the map site is a real OpenStreetMap stack, not a mock. Bring up your own WebArena map
   instance (site :3000, Nominatim :8085, OSRM :5000–5002; scripts in
   [webarena-setup](https://github.com/jayl940712/webarena-setup/tree/visualwebarena), `map/`) and put its
   address in `.env`: `CUA_GYM_WEBARENA_MAP_URL=http://<OSM_HOST>:3000` (replays) and
   `CUA_GYM_OSM_HOST=<OSM_HOST>` (read by `truth.py`; it overrides the host hard-coded in
   `scripts/_osm_common.py`). See [`TASK_MAP.md`](../../TASK_MAP.md).

The `TASK*.md` files at the repo root are the briefs earlier batches were written from; a new batch
usually starts as a new brief plus a spec.

## Run it

On the box that serves the mocks:

```bash
S=scripts/pipeline/specs/smk1.json
python3 scripts/webarena_datagen_pipeline.py --spec $S preflight      # 20 read-only checks
python3 scripts/webarena_datagen_pipeline.py --spec $S ping           # one tool-less session (~$0.03)
tmux new -s pipeline                                                  # survives disconnects
python3 scripts/webarena_datagen_pipeline.py --spec $S all            # the whole batch
python3 scripts/webarena_datagen_pipeline.py --spec $S status         # any time, from another shell
```

Single stages: `plan`, `author`, `gate`, `repair`, `verify`, `post`, `fix`, `export`, `truth`,
`diversity` (`--tasks <id>...` narrows gate/verify/repair). Everything is resumable: re-running a command skips work that is already done
(slots with bundles, tasks with a passing run). `--dry-run` exercises the control flow with no API calls
and no moves or copies.

## A new batch = a new spec

Copy `specs/smk1.json` (or `div1`–`div3`; the specs of our later batches are not published because they
embed internal evaluation data). The spec carries the batch id and directory, the model and effort, which docs
the agents read first (TASK file, contracts, corrections), the reference bundle, the gate flags, the
prior-batch roots for id reservation and near-dup triage, and the **slots**: one row per task
(site, skill pair, style). The prompts in `prompts/` are templates filled from the spec.

## Diversity batches (`specs/div1.json`, brief `TASK_DIVERSITY.md`)

A spec with `"diversity": true` and a `"plan"` section (instead of fixed `"slots"`) gets its slots from
`plan`, which measures the corpus (all six prior batches + every other pipeline batch) and gives each slot a
**novelty contract**: an official intent no corpus task resembles, or state fields no reward has ever
checked, plus unused seed-data entities and the site's over-used entities/families to avoid.
`scripts/pipeline/diversity.py` holds the model; the gate adds its checks (new family, no near-duplicate,
no over-used entity, target field checked, seed resembled not copied), and `diversity` writes
`pipeline/<batch>/DIVERSITY.md`: what the batch adds on each axis against the corpus baseline.

```bash
S=scripts/pipeline/specs/div1.json
python3 scripts/webarena_datagen_pipeline.py --spec $S plan          # review the 20 contracts (--force re-plans)
python3 scripts/webarena_datagen_pipeline.py --spec $S ping --role author
python3 scripts/webarena_datagen_pipeline.py --spec $S all
python3 scripts/webarena_datagen_pipeline.py --spec $S diversity     # report any time
```

## Per-role models

`"roles"` in a spec overrides the default profile per session role. `div1` authors with DeepSeek V4 Pro
through the NVIDIA inference API (Anthropic-compatible `/v1/messages`, as OSWorld-V2's dsv4 lanes do) and
verifies with Claude Opus on the CLI's own login; from `div3` both roles use the gateway (verify on
`aws/anthropic/bedrock-claude-opus-5-5`). The specs write the gateway as `"base_url":
"${CUAGYM_GATEWAY_BASE_URL}"`: put `CUAGYM_GATEWAY_BASE_URL=<gateway URL>` in `.env` (git-ignored, loaded at
start) or export it; an unset variable stops the driver. A gateway profile needs a bearer token, read from
`token_env` or `token_file` (default `~/.config/nvidia_inference_token`, mode 600); it is sent to the
gateway only, never printed, and the guard blocks sessions from reading it, `.env` or the CLI's credential
files. `preflight` sends one 16-token probe per gateway profile. `--default-profile-only` runs everything
on the default model (e.g. without a token).

Any role can use any model behind an Anthropic-compatible endpoint. For example, author with DeepSeek
and verify with either Claude or DeepSeek (the verify sessions write and audit the reward and replay the
golden path); the choice is yours.

## What it does per stage

| stage | how | output |
|---|---|---|
| preflight | endpoints, `/go` on each mock, pytest, SDK, CLI login, every file the spec names | `pipeline/<batch>/preflight.json` |
| author | one `task-author` session per site lane, ≤ 4 at once | `tasks/<site>/<id>/` |
| gate | `preflight_bundles.py`, `empty_state_probe.py`, `near_dup_scan.py` (triage only) | per-task pass/fail |
| repair | failed bundle → `rejected/<id>/`, fresh session re-authors the slot with the gate output; up to `--repair-rounds` | new bundle |
| verify | one `orchestrator` session per task (reward-gen, golden-browser, reward-audit inside), ≤ 5 at once; pass = `batch_orchestrator.task_is_complete()` | `runs/<id>/` |
| replace | a task that fails verification is rejected and its slot re-authored, gated and verified once (`--replace-rounds`) | |
| post | copy the verified files from `runs/<id>/` back into `tasks/<site>/<id>/` (backups in `pipeline/<batch>/pre_promote/`), re-gate, click-reachability | |
| fix | a verified task whose promoted bundle fails the post gate goes back to an orchestrator session with the gate's reasons; `post` re-runs (`--fix-rounds`) | |
| export | `export_nemo_rollouts.py`, filtered to tasks that cleared every stage; with `"provenance": true` also a provenance row per task | `<batch_dir>/<batch>_webarena.jsonl` |
| truth | retrofit: blind sessions write a `truth.py` per answer-scored task, `scripts/pipeline/truth.py` checks it, mismatches go to triage | `<batch_dir>/truth/report.jsonl` |

Why `post` promotes: the orchestrator repairs rewards in the run directory's copy of the bundle, but the
exporter reads `tasks/`. Without promotion a reward fixed during verification would ship unfixed.

## Code-computed truth

For answer-scored tasks the expected answer should come from code, not from a model. A slot with
`code_truth` must ship a `truth.py` that defines `compute(load)` (map: `compute(load, api)`) and
`render(answer)`; it reads mock data only through `load(app_dir, relpath)`, and map tasks only through
`api` (the site's own Nominatim and OSRM calls). The gate runs
`python3 scripts/pipeline/truth.py check <bundle_dir>`: the module must be reproducible, the task's reward
must pay 1.0 for the rendered answer, and give-up answers ("N/A", "") must pay below 0.5.

## Map batches

Set `"map": true` and `"external_base_url": {"webarena_map_mock": "http://<OSM_HOST>:3000"}` in the spec.
The map is stateless and off the hub, so the hub-only gate rules are waived for it, but every map
`nemo_reward.py` must still carry an unused `SID = "__CUA_GYM_SID__"` line or the exporter drops the task.

## Failure-grounded variants

`scripts/pipeline/assignments.py` turns a screen of an earlier batch (rollout rows, per-task pass rates,
gate report) into one assignment record per parent task with its failed rollouts and a failure label. A
spec whose `plan` has an `"assignments"` section gets one slot per (parent, variant): a harder or simpler
sibling, a fixed task, or a replacement for a rejected one.

## Guard hooks

Sessions run with `bypassPermissions` behind PreToolUse hooks (`bash_verdict`, `write_guard`):

* authors may write only `tasks/<site>/`; verifiers only `runs/<id>/`; everyone may use `/tmp`;
* `hub/`, `cuagym/`, `cua_gym_web/`, `scripts/`, `run_info/`, `prior_batches/`, `output/reference/`,
  `.claude/`, top-level docs and `.env` are protected;
* git is read-only; no `sudo`/`apt`; no env dumps or credential files; no `eval` or pipe-to-shell;
* the batch tooling and the mock lifecycle (`batch_orchestrator.py`, `export_nemo_rollouts.py`,
  `deploy-all.sh`, `start_webarena_mocks.sh`, `tmux kill`, `pkill`) are driver/operator-only;
* authors may not run validation code or Playwright (batch-5 rule 1) and cannot spawn sub-agents.

It is a seatbelt, not a sandbox: a Python script written under `/tmp` and then run is not inspected.
Denials are logged as `[guard] DENY` in the session's `transcript.md`.

## Files

```
<batch_dir>/pipeline/<batch>/
  state.json         slots, tasks, per-stage results, full history (resumable)
  stats/*.json       per invocation: tokens, turns, duration, cost per session
  sessions/<label>/  prompt.md, transcript.md, result.json for every session
  logs/*.log         every deterministic subprocess
  pre_promote/<id>/  bundle files overwritten by promotion
<batch_dir>/rejected/<id>/   bundle/, run/, REJECTED.md
```

## Known facts and open items

* **Pin the model.** Without `--model`, SDK sessions default to `claude-sonnet-4-6`, while an
  interactive `claude -p` defaulted to `claude-opus-5-5`. The spec's `model` is therefore required.
* **No background sub-agents.** In SDK (streaming) mode a turn that ends while background agents are
  still running produces a result, and closing the session kills them. The first smk1 run lost every
  orchestrator this way at round 0. Every session therefore gets `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`
  (sub-agents still run in parallel, in the foreground), and verify sessions get up to two in-session
  follow-ups when they stop before round 1 completes.
* Verification sub-agents (reward-gen, golden-browser, reward-audit) run inside the orchestrator's CLI
  process. Confirm from the first real run's transcripts that the hooks also fire for their tool calls.
* `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` is always set (batch 6: otherwise sub-agents die at 600 s
  with exit 0 and no verdict).
