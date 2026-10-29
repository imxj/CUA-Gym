This WebArena task already passed verification, but its final reward breaks the corpus gate, so it cannot ship. Repair the reward without changing what the task asks. Nobody will answer questions.

Task: $task_id
Run directory (yours; write only here): $run_dir
What ships is the canonical bundle at the run directory root: task.json, task_instruction.json, reward.py, nemo_reward.py, nemo_task.json, initial_setup.py when present, and golden_replay.py.

Gate failures:
$reasons

Do this, following .claude/agents/web-orchestrator.md for how agents are spawned and sandboxed:
1. Spawn reward-gen to rewrite reward.py and nemo_reward.py (one rubric, both files) so that every rule below holds, then re-inline nemo_reward.py (and initial_setup.py) into nemo_task.json byte for byte.
   - Read current_state only: never initial_state or state_diff (TASK4 S7). Compare against fixed seed values looked up in the mock's data (hub/websites/<app>/src/data), not against the episode's baseline.
   - No component may pay on an empty current_state or for inaction: collateral or "nothing else changed" checks count only once a positive marker of the real action is present.
   - A hard task declares `components` with partial credit. In a terse task no component name contains a preservation word (untouched, unchanged, kept, still, remains, retained, intact, ...).
2. Re-run the existing golden replay twice from fresh SIDs:
   python3 scripts/run_webarena_task.py $run_dir/task.json --replay $run_dir/golden_replay.py --output $run_dir/attempt-fix-<n> --mode legacy
   Both runs must score initial 0.0 and replay 1.0 with no browser errors.
3. Spawn reward-audit on a fresh audit sandbox (task.json, task_instruction.json, reward.py, nemo_reward.py, nemo_task.json, the new verification.json and lane evidence; no golden_replay.py) and copy its REVIEW.md to the run directory root. It must contain exactly "## Verdict: PASS".
4. Confirm the gate is clean for this task; both commands must print nothing for it:
   python3 scripts/preflight_bundles.py $runs_dir --quiet | grep -A12 "$task_id"
   python3 scripts/empty_state_probe.py $runs_dir --quiet | grep -A12 "$task_id"
$gate_waiver

Your final answer is parsed by a program. Return exactly the requested JSON object: {"task_id": "$task_id", "verified": <bool>, "initial_score": <number or null>, "replay_score": <number or null>, "rounds": <int>, "failed_conditions": ["<anything still failing>"], "notes": "<what you changed>"}.
