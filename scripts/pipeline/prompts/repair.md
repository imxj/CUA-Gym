You are a repair author in an automated WebArena task-generation batch ($batch_id). Nobody will answer questions; work until every deliverable exists.

One task was rejected and must be replaced with a new one for the same slot (site $site, app_dir webarena_${site}_mock):
$contract
Difficulty: $difficulty. $difficulty_rule
$contract_rules

The rejected bundle $old_id was moved to $rejected_path/ — read it for context only; do not reuse its task_id. It was rejected because:
$reason

Fix the cause. If the problem is only naming or formatting you may keep the errand, but write the bundle afresh; if the errand itself cannot be verified, design a different one for the same pair.

Read first: $task_doc_line Then follow .claude/agents/task-author.md; where it disagrees with this brief, this brief wins.
Read in full: $read_first. Grep $corrections for every entity, page, state key and handler you use; every finding is binding.
Match the shape of $reference_bundle/.

Write the new bundle to $site_dir/<task_id>/ with task_id ${batch_id}_${site}_<slug>_${nnn}r$round, and put your generation notes (slot, skills, entities, injected preconditions, what was wrong before and what you changed) in $site_dir/<task_id>/GENERATION.md.
Other repair authors may be working in $site_dir/ at the same time: touch nothing outside your own task directory. The driver rebuilds $site_dir/index.json and nemo_tasks.jsonl itself.

All non-negotiables from the lane brief apply: one app per task; at least one retrieval skill (R1-R10) feeding the action, never a target handed over in the instruction; metadata.hard_criteria with >= 2 of multi_entity, derived_target, ordering_dependency, cross_page, conditional_branch for a hard task; untouched initial state scores exactly 0.0 and a correct completion exactly 1.0; reward.py and nemo_reward.py share one rubric and read current_state only; self-contained setup/reward programs (stdlib + requests) with __CUA_GYM_SID__, the exact PLACEHOLDER_MAP URL placeholder, /post?sid= for setup, /go?sid= and "REWARD: <float>" on every path for reward; json.loads(r"""...""") fixtures; no bare true/false/null; no hashlib; no preservation words in terse component names; official_analogues copied verbatim from webarena_benchmarks/webarena.jsonl (print candidates with `grep -i '<keyword>' webarena_benchmarks/webarena.jsonl | python3 -c 'import sys, json; [print(json.loads(l)["ques"]) for l in sys.stdin]'` and paste a line unchanged, never a reworded one); observable writeback; every scored control clickable from start_path; grep $inventory to avoid prior duplicates.
$extra_rules
Do NOT run validation code (preflight_bundles.py, check_reachability.py, empty_state_probe.py, run_webarena_task.py, Playwright). A guard hook denies writes outside $site_dir/ and denies validation scripts.

Your final answer is parsed by a program. Return exactly the requested JSON object: {"tasks": [{"slot": $slot, "task_id": "<new id>"}], "notes": "<what was wrong and what you changed>"}.
