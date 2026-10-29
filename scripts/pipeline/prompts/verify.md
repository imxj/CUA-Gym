Validate this imported WebArena task with the browser-only pipeline.

Task manifest: $task_json
Endpoint registry: (environment variables)
Output directory: $run_dir
Session mode: $mode

Execute the complete workflow from .claude/agents/web-orchestrator.md. Do not create VMs and do not ask clarifying questions.
The task bundle under $bundle_dir/ is read-only to you: copy it into the output directory (workflow step 6) and make every repair there. A guard hook denies writes anywhere except the output directory and temporary files.

Whatever reward the rounds settle on must also pass the corpus gate, or the task is discarded after verification. Tell reward-gen these rules up front:
- read current_state only, never initial_state or state_diff (TASK4 S7); compare against fixed seed values from the mock's data instead;
- no component pays on an empty current_state or for inaction; collateral checks count only once a positive marker of the real action exists;
- a hard task declares `components` with partial credit; terse tasks have no component named for preservation.
Before you finish, both of these must print nothing for this task:
  python3 scripts/preflight_bundles.py $runs_dir --quiet | grep -A12 "$task_id"
  python3 scripts/empty_state_probe.py $runs_dir --quiet | grep -A12 "$task_id"
$gate_waiver

When the workflow finishes, your final answer is parsed by a program. Return exactly the requested JSON object: {"task_id": "<id>", "verified": <bool>, "initial_score": <number or null>, "replay_score": <number or null>, "rounds": <int>, "failed_conditions": ["<agreement condition that failed>", ...], "notes": "<one paragraph>"}.
