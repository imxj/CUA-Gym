# TASK_SMOKE.md — 10-task smoke batch (`smk1`)

A pipeline check, not a production batch: author 10 new WebArena tasks, verify
all 10 end to end, and export them as NeMo rollout rows. If this comes out
clean, the same box can run a full batch from a TASK file written the same way.

Do not ask clarifying questions; the decisions below are made.

---

## 1. Spec layering — read in this order

This batch follows **batch 6**, the most recent WebArena batch (600/600
verified). Batch 6 is defined as a stack of overrides; read all of it:

1. `run_info/b4__TASK4.md` — base contract: style (S3), start_path (S5),
   click-reachability (S6), ground rules (S7), definition of done (S9).
2. `run_info/b5__BATCH5_CONTRACT.md` — skills, injection, composition rules.
3. `run_info/b6__BATCH6_CONTRACT.md` — medium == exactly one R + one A, and
   **the 13 rules that cost the most** (carry them into every author prompt).
4. `docs/SKILL_TAXONOMY.md` — the R1–R10 / A1–A13 codes.
5. `run_info/b5__CORRECTIONS.md` (findings 1–122) and
   `run_info/b6map__CORRECTIONS.md` (B6-1… plus MAP-*) — every finding is binding.

Where this file and those disagree, **this file wins**.

## 2. Environment (set up 2026-10-02 on `testvm1`; verify, don't redo)

| Thing | Where |
|---|---|
| Repo | `~/CUA-Gym` @ `harness/portable-setup`, hub @ `594c7b1c` |
| Mocks | tmux `cua-gym-hub`, windows 0–4 = ports 8000–8004 (classifieds, gitlab, reddit, shopping_admin, shopping). Restart with `~/start_webarena_mocks.sh`. **Do not use `hub/deploy-all.sh`** — at `594c7b1c` it serves all 103 mocks and the WebArena ones land on 8088–8092. |
| Endpoints | `.env` → `http://localhost:800X/`; map → `http://18.116.12.228:3000` |
| Inspiration corpus | `webarena_benchmarks/webarena.jsonl` (symlink to `~/webarena_benchmarks`) |
| Reference bundle | `output/reference/bestseller_price_bump_autumn_2022_window_up_8pct_006/` — verified (initial 0.0, replay 1.0, REVIEW PASS, reachability PASS). The shape every bundle must match. |
| NeMo row example | `cuagym/data/example.jsonl` (the reference task's exported row) |
| Prior tasks (2,752) | `prior_batches/webarena_*.jsonl` (exported rows of all six batches); `prior_batches/inventory/<site>.tsv` (batch, task_id, intent); `prior_batches/all/<site>/<task_id>/` (stubs for `--prior-batch` and `near_dup_scan.py`) |

## 3. The batch

**10 tasks, all `medium`: exactly two skills, one retrieval feeding one action.**
Each slot below fixes the site, the (R, A) pair and the style. The author picks
the entities and the errand; the pair is the unit of assignment (batch 6
override 2).

| slot | site | pair | style |
|---|---|---|---|
| 1 | gitlab | R1 → A1 | terse |
| 2 | gitlab | R7 → A3 | terse |
| 3 | gitlab | R5 → A10 | explicit |
| 4 | reddit | R4 → A8 | terse |
| 5 | reddit | R3 → A7 | explicit |
| 6 | shopping | R9 → A11 | terse |
| 7 | shopping | R1 → A2 | terse |
| 8 | shopping_admin | R10 → A7 | explicit |
| 9 | shopping_admin | R2 → A4 | terse |
| 10 | shopping_admin | R6 → A10 | terse |

If a pair is genuinely infeasible on its site after reading the source, swap it
for a different pair on the same site, keep the style, and record why in
`GENERATION.md`. Never pad to three skills; never relabel.

Constraints, all from batches 4–6:

* **Terse** intents ≤ 40 words. 7 terse / 3 explicit overall, as tabled.
* **`start_path: "/"`** for at least 8 of 10 (batch 6 ran 96%).
* **IDs:** `smk1_<site>_<slug>_<NNN>`; directory name == `task_id`. Must not
  collide with any id in `prior_batches/all/`.
* **No re-authoring a prior task with a value swapped.** Grep
  `prior_batches/inventory/<site>.tsv` for the entities and verbs you plan to
  use before writing. Resembling an official benchmark task is fine;
  reproducing one verbatim is not.
* **Metadata** (batch 5 override 3 + batch 6): `style`, `difficulty: "medium"`,
  `skills: [R?, A?]`, `skill_chain`, `official_analogues` (verbatim intents
  from `webarena_benchmarks/webarena.jsonl` — the gate cross-checks),
  `injected_preconditions`, `derived_from` (prior task id or `null`),
  `inspiration_ids`.
* Injection via `initial_setup.py` is allowed and encouraged as a design tool
  (batch 5). Check the mock's `set` semantics first: reddit **replaces**, shopping
  and shopping_admin shallow-merge, gitlab unmeasured.
* Rewards read `current_state` only; fixtures as `json.loads(r"""...""")`; no
  `hashlib`; allowed reward imports only.
* **Authors run no validation code.** Write the bundles and stop.

## 4. Phases

### Phase 1 — Preflight
Ports 8000–8004 answer; `GET /go?sid=preflight` returns `state_diff == {}`;
`.env` resolves all five apps; `python3 -m pytest tests/ -q` → 24 passed;
`claude auth status` → logged in. Record in `output/PROGRESS.md`.

### Phase 2 — Author (4 lanes in parallel, one `task-author` per site)
Output `output/tasks/<site>/<task_id>/` with the canonical files
(`task_instruction.json`, `task.json`, `reward.py`, `nemo_reward.py`,
`nemo_task.json`, `initial_setup.py` when setup is non-null), plus
`output/tasks/<site>/{index.json,GENERATION.md,nemo_tasks.jsonl}`.

### Phase 2b — Gate (orchestrator runs these, not authors)
```bash
python3 scripts/preflight_bundles.py output/tasks/<site> --medium-is-exactly-two \
    --prior-batch prior_batches/all --quiet
python3 scripts/empty_state_probe.py output/tasks --quiet
python3 scripts/near_dup_scan.py --new output/tasks/<site> --prior prior_batches/all/<site>
```
A failing bundle goes back to a fresh `task-author` for that slot with the gate
output. Do not hand-patch. Flagged near-dups are read and judged, not
auto-rejected.

### Phase 3 — Verify
```bash
CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 python3 scripts/batch_orchestrator.py \
    output/tasks/gitlab output/tasks/reddit output/tasks/shopping output/tasks/shopping_admin \
    --output output/runs -c 5 --timeout 60 --max-turns 220 --dangerously-skip-permissions
```
The env var is mandatory (batch 6: without it sub-agents die at 600 s with exit
0 and no verdict). Then:
```bash
python3 scripts/check_reachability.py output/runs --quiet
python3 scripts/_b5_status.py --runs output/runs --failures
python3 scripts/detect_flaky.py
```
A task that fails all three orchestrator rounds is replaced by a newly authored
task for the same slot — one replacement round, then report.

### Phase 4 — Export
```bash
python3 scripts/export_nemo_rollouts.py output --output output/smk1_webarena_10.jsonl
```

## 5. Definition of done

`output/smk1_webarena_10.jsonl` holds 10 rows, every one from a bundle with a
passing `verification.json`, `## Verdict: PASS`, a clean reachability check,
agreeing attempts, and an empty-state score of 0.0. Report per-slot status,
replacements, and any systemic issue. If fewer than 10 make it, say so plainly
with the blocker per slot.
