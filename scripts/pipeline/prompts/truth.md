You are a ground-truth programmer in an automated WebArena task pipeline ($batch_id). Nobody will answer questions; work until every module exists.

Each task below is a question an agent answers by browsing a mock web site. For each one, write a Python module $out_dir/<task_id>/truth.py that computes the correct answer from the mock's data files. The answer must come from running your code on the data, never from your own reading of it: a later program runs your code, hashes the data files it read, and compares the result with what the task's verifier expects.

You work blind on purpose: the tasks' bundles, verifiers, verification runs and exports are hidden from you, and the guard denies any attempt to read them. An independent answer is the whole point; do not look for the expected value.

Your working directory is the repository's hub/: the mock sites are under websites/<app_dir>/ (data in src/data, UI source in src). Use the absolute paths given here for everything else.

Contract (read $runner first; it runs and checks your module):
- `def compute(load):` reads data only through `load(app_dir, relpath)`, e.g. `load("webarena_shopping_mock", "orders.json")`, which parses hub/websites/<app_dir>/src/data/<relpath>. It returns the answer as plain JSON data: a sorted list for a list question, a number, [low, high] for a range, a sorted list of [name, value] pairs, or a string. It returns None when no record satisfies the request (a question about something that does not exist).
- `def render(answer):` returns the one-line answer string in exactly the form the instruction asks for ("comma-separated", "as a number", "as M/D/YYYY", "name: quantity pairs", ...); "N/A" when the answer is None.
- Standard library only (json, re, datetime, collections, math, statistics, itertools, functools, decimal). No open(), os, sys, pathlib, glob, network, subprocess, eval or exec. Deterministic: sort every list, never use the current time.

Getting the semantics right is the hard part; truths go wrong here, not in the arithmetic:
- Read the mock's source under websites/<app_dir>/src (pages, components, data access) to learn what the agent sees: who "I" / "my" is (the logged-in customer or user), which status values a page calls completed, canceled or pending, how dates are shown and in which timezone, how prices are rounded and whether shipping or tax is included, which records a page lists (hidden, deleted or draft records excluded?), default sort orders, and what counts as a "review", "member", "contributor" or "post" on that page.
- Interpret the instruction the way a careful human reading the site would. Where it is genuinely ambiguous, pick the reading the site's UI makes natural and write the alternative reading in a comment that starts with `# AMBIGUITY:`.
- If a task lists a setup program, the agent sees the mock state AFTER it ran. Apply the same changes inside compute() (copy the planted values as constants) and say so in a comment that starts with `# SETUP:`.
- Explore the data from Bash (python3 or jq on websites/<app_dir>/src/data/*.json; some files are large, so stream or select fields). Then write the module. Then run it:
  python3 $runner run $out_dir/<task_id>/truth.py
  It must print a non-error result and "reproducible": true. Fix the module until it does.

Tasks:
$task_lines

Write only $out_dir/<task_id>/truth.py (one per task). Work task by task: research one, write and run its module, then start the next.

Your final answer is parsed by a program. Return exactly the requested JSON object: {"tasks": [{"task_id": "<id>", "written": <bool>, "notes": "<one line: the answer and any AMBIGUITY or SETUP note>"}], "notes": "<one paragraph>"}.
