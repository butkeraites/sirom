# ADR-0003: Count cluster nodes from the tree, not via stage-tagged results

- Status: Accepted (2026-06-05), implemented
- Context: architecture review finding #4 (sharpened on adversarial verification)

## Decision

Do **not** introduce stage tags or a results-partitioning seam on
`ProblemsBucket.results`. Instead, source the cluster-node count directly from
`bucket.cluster_tree.get_all_nodes()` and keep the cheap `results[:n]` prefix for the
scenario-optimal count. (Implemented in the same change as this record.)

## Context

`ProblemsBucket.results` is a flat list to which `solve()` appends scenario solves and
`solve_cluster_tree()` appends cluster re-solves. A review candidate proposed exposing
named, stage-tagged groups (e.g. `scenario_results` / `cluster_results`) so the stage
boundary lived in the interface.

## Why this was sharpened rather than implemented as proposed

- **The slice was in exactly one place.** `results[:n]` / `results[n:]` appeared only
  in `api/service.solve_problem`, feeding two summary counts — not "two places".
- **One of the counts was already available.** `cluster_nodes` equalled
  `len(bucket.cluster_tree.get_all_nodes())`, which the bucket already holds.
- A general stage-tagged-result abstraction would be speculative generality for a
  single call site — a seam nothing varies across.

## What was done instead

- Deleted the `results[n_scenarios:]` tail slice and its temporaries.
- `scenarios_optimal` keeps the order-stable `results[:n]` prefix.
- `cluster_nodes` is read from the tree (None-guarded) — equal today (one re-solve per
  node) but robust to future append-order changes.

## Consequence

The stage boundary is not encoded as a new interface; the bucket is asked what it built.
If results consumption ever needs more than two counts, revisit — but not before.
