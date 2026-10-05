You are lane $batch_id-$site of an automated WebArena task-generation batch. Nobody will answer questions; work until every deliverable exists.

Read first: $task_doc_line Then follow .claude/agents/task-author.md; where it disagrees with this brief, this brief wins.
Read in full before designing anything: $read_first.
Then grep $corrections for every entity, page, state key and handler you plan to use on $site. Every finding there is binding.
Match the file set and shape of $reference_bundle/ (task.json, task_instruction.json, reward.py, nemo_reward.py, nemo_task.json, and initial_setup.py when setup is non-null). cuagym/data/example.jsonl is a real exported row.

Write $n new RL tasks for site $site (app_dir webarena_${site}_mock), exactly one per slot:
$slot_lines
Difficulty: $difficulty. $difficulty_rule
$contract_rules
Output directory: $site_dir/ — one directory per task; directory name == task_id.
Task id format: ${batch_id}_${site}_<slug>_<NNN>, where NNN is the slot number zero-padded to three digits.

Non-negotiables, restated because they are what break downstream:
- One app per task.
- Every task derives its target from the site: metadata.skills includes at least one retrieval skill (R1-R10) feeding the action. A one-step instruction that hands over the target ("set my status to X") fails the gate; make the agent find it first ("set my status to the name of the project I starred most recently").
- A hard task carries metadata.hard_criteria with at least two of: multi_entity, derived_target, ordering_dependency, cross_page, conditional_branch.
- The untouched initial state (after initial_setup.py runs) scores exactly 0.0; a correct completion scores exactly 1.0; partial credit via components.
- reward.py and nemo_reward.py implement the same rubric and read current_state only, never state_diff or initial_state.
- Setup and reward are self-contained single programs using only the standard library plus requests; they contain __CUA_GYM_SID__ and the exact __CUA_GYM_WEBARENA_<APP>_URL__ placeholder from PLACEHOLDER_MAP in cuagym/hub_apps.py; setup POSTs to /post?sid=, reward GETs /go?sid= and prints "REWARD: <float>" on every output path including its error path.
- Inline fixtures as json.loads(r"""..."""); no bare true/false/null in Python source; no hashlib.
- In terse tasks, no component name may contain a preservation word (untouched, unchanged, kept, still, remains, retained, intact, ...): the gate rejects those by name even when the logic is a positive check.
- Benchmark rows are inspiration only (sample with scripts/sample_webarena_inspirations.py --site $site); never reproduce an official intent verbatim. metadata.official_analogues must quote real intents verbatim: copy them, never write them. Print candidates with `grep -i '<keyword>' webarena_benchmarks/webarena.jsonl | python3 -c 'import sys, json; [print(json.loads(l)["ques"]) for l in sys.stdin]'` and paste a printed line unchanged. A generalised or reworded intent ("... about a specific topic") is not in the benchmark and fails the gate; this was the most common rejection in this batch.
- A retrieval must end in an observable writeback; no answer-only tasks.
- Every scored control must be reachable by clicking from start_path. Prefer start_path "/".
- Before writing each task, grep $inventory for its entities and verbs: no task_id collision with prior batches, and no value-swapped re-authoring of a prior task. Record the closest prior task in metadata.derived_from (or null).
- Do NOT run validation code (preflight_bundles.py, check_reachability.py, empty_state_probe.py, run_webarena_task.py, Playwright). Verification is a later phase with its own agents. Use Bash only to read source, seed data, the benchmark and the inventories.
$extra_rules
A guard hook denies writes outside $site_dir/ and denies validation scripts; a denial means you are off-brief, not that you should find another way.

Work slot by slot: research one slot, write its complete bundle to disk, then start the next. Never research every slot before writing anything. Your session has turn and time limits; bundles already on disk survive a cutoff, designs you have not written do not.

Other author sessions may be writing other slots into $site_dir/ at the same time: touch only your own task directories. For each task write its generation notes to $site_dir/<task_id>/GENERATION.md (slot, skills, entities, injected preconditions and why, derived_from, inspiration ids, rejected ideas). Do not write $site_dir/index.json or $site_dir/nemo_tasks.jsonl: the driver rebuilds both from the bundles.

Your final answer is parsed by a program. Return exactly the requested JSON object: {"tasks": [{"slot": <int>, "task_id": "<id>"}, ...], "notes": "<one paragraph>"}.
