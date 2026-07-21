# Pass@k: trained drafts vs teacher, and prob vs jsd (real ckpt_best eval)

## Formula

`pass@k = 1 - C(n-c, k) / C(n, k)` (Chen et al. 2021), `n=64` samples/prompt,
`c` correct among them, averaged over 100 held-out prompts. At `k=n=64` this
is just "did ≥1 of 64 tries succeed" per prompt.

33 checkpoints, real `ckpt_best` offline eval:
[results/passK/passk_hparam_sweep.csv](../results/passK/passk_hparam_sweep.csv).
Noise floor = 2×std(pass@k) across the flat low-LR `prob` cluster, per k per
dataset — a delta only counts as real if it clears this.

**What "clears the floor" means:** most `prob` configs sit numerically
*above* jsd's curve on every dataset, including `math_eval` — the direction
is consistently prob-favoring. "Clears the floor" means that gap is bigger
than the run-to-run spread already measured across a set of nominally
similar low-LR configs (2×std of that spread). Where a gap does *not*
clear it, prob is still numerically ahead, just by less than what a single
seed's own noise could produce on its own — so it isn't confirmed as a
real effect, not "prob is worse."

**What `best_val_block_eff` is, and isn't:** it's a training-time metric
only — computed once per checkpoint during training, on `math_val`, via
the block-efficiency verifier. It was never computed on `math_eval`,
`gsm8k_eval`, or `olympiad_eval` — those three are held-out pass@k sets
only, with no verifier/block_eff run on them at all. Every "BE" number
below is that single training-time value; comparing it against a
checkpoint's pass@k on any of the three held-out sets is comparing two
measurements from two different data sources, not "BE on that dataset."

## Findings

- Cases where a trained checkpoint's pass@64 dips below the untrained
  baseline are explained in the footnote.

- Checked against jsd on pass@k, per dataset. Two different questions,
  not the same thing: **direction** (of 33 checkpoints, how many sit
  numerically above jsd at this k) vs **floor-clearing** (whether that gap
  is big enough to call a real effect, not just noise). Direction turns
  out to be overwhelmingly prob-favoring on *every* dataset — the
  difference between datasets is entirely in whether that direction
  survives a floor.

  - `math_eval`: **26–31 of 33** sit above jsd at k1–k32 (drops to 18/28 at
    k64). Despite that consistent direction, **nothing clears a real
    floor** at any k — the per-config gaps are individually too small.

    ![pass@k, every checkpoint, math_eval](../results/passK/passk_curves_all_math_eval.png)

    Best-prob (`ceanneal1500_lr1e-5_wu20`) tracks just above best-jsd
    (`jsd_lr3e-6_wu10_lrmin0.1_wd0.01`) the whole curve, and most thin red
    lines sit above the thin blue lines too — but the two families'
    spreads overlap enough that no single gap clears a real floor.

  - `olympiad_eval`: **27–32 of 33** sit above jsd at k1–k32 (drops to
    18/29 at k64) — just as consistent a direction as `math_eval`, if not
    more so. Same result as `math_eval` though: only 2/33 gaps are large
    enough to individually clear the floor.

    ![pass@k, every checkpoint, olympiad_eval](../results/passK/passk_curves_all_olympiad_eval.png)

    Best-prob (`lr1e-5_wu5_lrmin0.1_wd0.01`) edges above best-jsd, and
    (matching the count above) most thin red lines sit above thin blue —
    the chart's visual impression of "a mixed picture" undersells how
    consistent the raw direction actually is here; it's the *floor*, not
    the direction, that most configs don't clear.

  - `gsm8k_eval`: **14–21/33** clearly win above noise, concentrated at
    k16/k32/k64 (0/33 at k1–k4) — the one held-out set with a real
    pattern. Each of these
    checkpoints' *training-time* `best_val_block_eff` (measured on
    `math_val`, unrelated to `gsm8k_eval`) sits 0.18–0.46 below jsd's
    5.994 — the checkpoints that do best on this held-out pass@k set were
    not the ones jsd's own training-time metric would have picked.

    ![pass@k, every checkpoint, gsm8k_eval](../results/passK/passk_curves_all_gsm8k_eval.png)

    Different from the other two: best-prob (`po_prob_lr7e-6_wu20`) sits
    clearly above best-jsd across nearly the entire curve, and the
    red/blue thin-line clusters visibly separate rather than overlap.

  All three charts use best-of-family by pass@k gap-closed (a different,
  hindsight selection criterion than the BE-deployed-pick comparison
  above) — a complementary view, not a contradiction.

- **Multi-root (N=16, N=16+random-offset, N=32) consistently beats jsd on
  `gsm8k` k32/k64** — all three tested clear the floor at both k32 and k64
  (deltas 0.022–0.04 vs floor 0.021–0.028). Plain N=16 (no offset) has the
  broadest effect, also clearing k16 (Δ=0.031). **Winner: yes, consistent
  across the whole N-family tested.** `N=8` is training now (not yet
  through `passk_eval.py`) — add once its `ckpt_best` is evaluated.

- **Grad_clip does NOT show a consistent pass@k effect** — `gradclip100`
  clears `gsm8k` k16/k32/k64 (Δ 0.033–0.04); `gradclip10` and `gradclip1000`
  "clear nothing" meaning their gap to jsd stays inside the noise floor at
  every k (gradclip10's closest miss is k32, Δ=0.020 vs floor 0.021), not
  that they fall below jsd. The default (`gradclip=1.0`) was never run
  through `passk_eval.py`, so it can't be included in this comparison.
  **No consistent winner across the 3 clip values tested** — don't
  generalize from the `gradclip100` result alone.

- **Prefix_M family does NOT show a consistent pass@k effect either**, now
  that `M4`/`M8`/`M16` (warm-started)/`M16_cold` are all in the
  offline-eval set. `M4` clears `gsm8k` k16/k32/k64 (Δ 0.021–0.030), `M8`
  clears the same three (Δ 0.030–0.037), `M16` warm-started also clears
  them (Δ 0.030–0.032), but `M16_cold` (same config, clean cold-start redo
  of the warm-started run) doesn't clear anywhere (Δ 0.010–0.021, inside
  the floor). No trend with `M` itself — it's the same already-known
  gsm8k pattern showing up in most (not all) `prob` checkpoints
  regardless of family.

- **Teacher temp shows the same isolated, non-lever pattern**: `ttemp=0.5`
  clears `gsm8k` k16/k32/k64 (Δ 0.027–0.033), `ttemp=0.7` stays inside the
  floor everywhere. Default is `ttemp=1.0`, not evaluated through
  `passk_eval.py` (same gap as the grad_clip default above). One point in
  a two-point family clearing isn't a confirmed temp effect, same caveat
  as `gradclip100` above.

- **CE-anneal (aux_weight=0.5) at lr=3e-6 tracks its own no-anneal base**,
  not a new effect: both clear `gsm8k` k32 (anneal Δ=0.028, base Δ=0.021),
  anneal additionally clears k16 (Δ=0.033) where the base doesn't. Anneal
  doesn't move the needle much at this LR — unlike lr=1e-5, where
  `ceanneal3000_auxw0.5` is prob's actual best-BE deployment pick.

## Footnote — why one checkpoint's trained draft is below the untrained student

[passk_by_checkpoint_math_eval_06b_8b.png](../results/passK/passk_by_checkpoint_math_eval_06b_8b.png):
`warm_anneal_lr1e5` and `jsd_mathhard_s123` beat the untrained `Qwen3-0.6B`
at `k=1` (0.178/0.153 vs 0.119) but the untrained curve is steeper and
crosses back above both by `k≈16-32`, finishing higher at `k=64`
(0.570 vs 0.530/0.510). `lr5e6` and `ce_lr1e5` stay above baseline the whole
curve on the same dataset — this crossover is checkpoint-specific, not a
property of training in general. [passk_by_model_size_all_datasets.png](../results/passK/passk_by_model_size_all_datasets.png)
is the control: across model *size* (no training), every curve is strictly
monotonic at every k — confirms the pass@k math itself is correct.

## Dataset headroom (k=64, untrained 0.6B → teacher 8B)

| dataset | k=64, 0.6B→8B | note |
|---|---|---|
| `olympiad_eval` | 0.240 → 0.330 | lowest absolute scores everywhere (32B teacher itself only hits 0.36) |
| `math_eval` | 0.570 → 0.630 | mid-range, not saturated — but nothing clearly wins above noise here regardless |
| `gsm8k_eval` | 0.920 → 0.990 | near-ceiling by absolute level, yet the dataset where a real prob-vs-jsd pattern actually shows up |

## Practical read

- Judge a checkpoint by pass@k gap-closed to the teacher, per dataset —
  not by pass@1 or by training-time block_eff alone; block_eff is measured
  on `math_val` only and is not a substitute for held-out pass@k.
- `math_eval` and `olympiad_eval` pass@k differences are not currently
  usable as a go/no-go signal — direction favors prob, but nothing clearly
  wins above noise there.
- `gsm8k_eval` at k16/k32/k64 is currently the most useful lens for a real
  prob-vs-jsd gap.
- Re-derive the floor whenever the checkpoint set changes; it moves as new
  runs join the reference cluster.
