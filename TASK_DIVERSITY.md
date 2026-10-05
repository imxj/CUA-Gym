# TASK_DIVERSITY.md — batches whose job is to widen the corpus

The corpus already holds 2,752 verified WebArena tasks. They look varied (almost every id is unique) but
collapse into ~331 effective task shapes: 179 families of 10+ near-identical tasks hold 73% of the corpus,
38% of GitLab tasks name the same ten projects, and only 7–23% of official WebArena intents have a
lexically similar generated task. A new batch is worth running only if it adds what the corpus lacks.

## What every slot gives you

Each slot carries a **novelty contract**, planned by the pipeline from measured gaps:

* **inspiration** — an official WebArena intent no corpus task resembles. Build the same *kind* of
  everyday errand on this mock, with different entities and values. Official intents are short (mean ~18
  words); a slot marked *short* wants ≤ 20 words.
* **target state** — fields of the mock's state no reward has ever checked. The reward must check at least
  one of them; find the UI control that writes it before designing anything.
* **open** — a skill pair the corpus is thin on (cross-entity joins, semantic matching, conditional actions,
  arithmetic edits, attribute matrices, ...). Invent a task shape no prior task has, using exactly that pair.
* **unused entities** — seed-data names no prior instruction mentions. Prefer them (or any other unused one).
* **do not name** / **over-used shapes** — the corpus's ten most-named entities and twelve biggest
  families on this site.

## What "new" means — the gate checks all of it

| check | fails when |
|---|---|
| family | `metadata.family` repeats an existing shape (token overlap ≥ 0.67 with any corpus family), or another task in this batch already has it |
| near-duplicate | the instruction is within token-Jaccard 0.55 of any corpus task, or of an earlier slot in this batch |
| same template | with names, numbers, paths and quoted text blanked, the instruction is within Jaccard 0.55 of a corpus task or an earlier slot in this batch: "In repo A, comment X on the open issue whose description mentions Y" and the same sentence for repo B is one task, not two |
| entities | the instruction names a "do not name" entity |
| target state | a target-state slot's reward names none of its target fields |
| inspiration | the instruction is within Jaccard 0.75 of its official seed (a copy, not a resemblance) |
| short | a short slot's instruction exceeds ~20 words |

Plus every existing gate (bundle schema, terse rules, empty-state 0.0) and full verification.

## Required metadata (in addition to batches 4–6)

`slot`, `family` (2–6 token snake_case slug naming the shape, not the entity), `entities`,
`targeted_fields`, `seed_id` (inspiration slots), `skills`, `difficulty`, `style`.

## Keep it solvable

Diversity is the goal; difficulty is secondary. Derive difficulty from the skill chain (batch-5 rule), keep
the golden path within ~25 UI actions (the policy has a 50-step budget and 67–69% of its failures hit it),
and keep the reward to at most 3 components.

Everything in TASK4 / BATCH5_CONTRACT / BATCH6_CONTRACT and both CORRECTIONS files still applies,
except batch 6's all-medium override.
