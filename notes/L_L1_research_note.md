# Tree depth (L) and delayed branching (L1) — research note

0.6B/8B, `math_hard`, K=3, all 3 eval datasets (`math_eval`, `gsm8k_eval`,
`olympiad_eval`), single seed. Combines every L/L1 eval grid in the repo:
[results/jsd_enrich_ddte_results.csv](../results/jsd_enrich_ddte_results.csv) (L=8, math_eval + ~1 gsm8k_eval row),
[per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv](../results/per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv) (same jsd checkpoint, L=8, fills in `olympiad_eval` — averaged over 4 repeat-run timestamps),
[l_sweep_jsd_rerun.csv](../results/passK/l_sweep_jsd_rerun.csv) / [l_sweep_jsd_enrich_final.csv](../results/passK/l_sweep_jsd_enrich_final.csv) (L=16/32, L1=0, all 9 verifiers, confirmatory reruns, all 3 datasets complete),
[l1_sweep_jsd_rerun.csv](../results/passK/l1_sweep_jsd_rerun.csv) / [l1_sweep_jsd_enrich_final.csv](../results/passK/l1_sweep_jsd_enrich_final.csv) (L=16/32, L1 sweep, traversal+specinfer, all 3 datasets complete).
Script: [scripts/plot_L_L1_analysis.py](../scripts/plot_L_L1_analysis.py), one PNG per figure per dataset.

**Caveat used throughout:** the L=8 checkpoint (`jsd_mathhard_s123`) is a
different training run than the L=16/32 checkpoint
(`jsd_lr1e-5_wu10_lrmin0.1_wd0.01`) — same loss/dataset, different
seed/hyperparams. Trend direction across L=8→16→32 is still meaningful;
exact magnitudes shouldn't be compared as if it were one continuous sweep.
`olympiad_eval` at L=8 exists **only for jsd** (via the per_checkpoint_sweeps
file above, a short L1=0/3/4/5 sweep, averaged over repeat runs) — enrich's
exact checkpoint (`jsd_flat_enrich_K3_mathhard_s123`) has no matching
`olympiad_eval` or `gsm8k_eval` L=8 data anywhere in the repo (checked:
several *other* enrich variants — different K, different seed, `temp07`,
`draftcond`, a different model pair — do have olympiad_eval L=8 rows, but
none is the same checkpoint, so none is used here). Enrich's L=8 stays
`math_eval`-only; its `olympiad_eval`/`gsm8k_eval` charts start at L=16.
Prose below describes `math_eval`; the same pattern holds on `gsm8k_eval`
and `olympiad_eval` unless noted — see the linked charts for each.

## Bottom line, if the question is "does this speed up inference"

Modest but real, and how big the win looks depends entirely on what it's
compared against:

- **Vs. the untuned default at the same depth** (L=16, L1=0, what would
  otherwise be deployed): the best config found (jsd, traversal, L=16,
  L1=10) is **+10.2% faster** (11.66 → 12.84 tok/s). Real, single run each
  side — treat as directional, not certain, until repeated.
- **Vs. going to L=32 for more BE** (L1=0): **+72.9% faster** (7.43 → 12.84
  tok/s). The bigger, more confident number, but it's mostly the value of
  *not making a mistake* (don't chase BE by going deeper) rather than an
  improvement over an already-decent L=16 baseline.
- **What this note cannot answer:** none of the CSVs here have a plain
  autoregressive (no speculative decoding at all) throughput number, so
  there's no "Nx faster than no speculation" figure available — only
  relative comparisons among the speculative-decoding configs actually
  tested.

So: yes, L1 tuning speeds up inference over the untuned default, by a real
but single-digit-to-low-teens percent, and it prevents a much larger loss
that chasing BE via L=32 would cost. Frame it as closing a gap already on
the table, not a multiplier-scale breakthrough.

## 1. Raising L trades block efficiency for throughput, for every verifier

BE and throughput vs L, all 9 verifiers:
[math_eval](../results/passK/L_L1_fig1_be_throughput_vs_L_all_verifiers_math_eval.png) ·
[gsm8k_eval](../results/passK/L_L1_fig1_be_throughput_vs_L_all_verifiers_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/L_L1_fig1_be_throughput_vs_L_all_verifiers_olympiad_eval.png)

- BE rises with L for all 9 verifiers, L1=0 (full branching). `bv` and
  `traversal` gain the most (5.8→10.0, 5.9→8.9); `nss`/`max`/`khisti` barely
  move at all.
- Throughput **falls monotonically with L for every single verifier**, no
  exceptions. The best-BE verifiers at L=32 (`bv`, `traversal`) are also
  among the fastest-falling on throughput.
- The best/worst verifier gap on BE widens with L (bv−nss: 1.9 at L=8 → 4.9
  at L=32) — depth rewards verifiers that were already good, not the weak
  ones.

## 2. L1 (delayed branching) breaks that tradeoff — BE and throughput both rise together

traversal, BE and throughput vs L1:
[math_eval](../results/passK/L_L1_fig2_be_throughput_vs_L1_traversal_math_eval.png) ·
[gsm8k_eval](../results/passK/L_L1_fig2_be_throughput_vs_L1_traversal_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/L_L1_fig2_be_throughput_vs_L1_traversal_olympiad_eval.png)

specinfer, BE and throughput vs L1:
[math_eval](../results/passK/L_L1_fig3_be_throughput_vs_L1_specinfer_math_eval.png) ·
[gsm8k_eval](../results/passK/L_L1_fig3_be_throughput_vs_L1_specinfer_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/L_L1_fig3_be_throughput_vs_L1_specinfer_olympiad_eval.png)

- At L=16 and L=32, increasing L1 from 0 raises **both** BE and throughput
  for `traversal` and `specinfer` — the opposite of what raising L alone
  does.
- `specinfer` gains far more than `traversal` (BE +2.4 vs +1.5 from L1=0 to
  L1=12 at L=32), consistent with the earlier DDTE finding that top-down (OT)
  verifiers waste more branching in the shallow zone than bottom-up
  (traversal) ones — delayed branching has more to recover there.
- L=8 is too noisy to read an L1 trend from (throughput swings 6→15 tok/s
  between adjacent L1 values, single-run variance dominates at this small a
  unit cost) — don't use the L=8 points for tuning L1, only L=16/32.

## 3. L=16 with L1 tuned beats L=32 outright, not just "faster but worse"

Efficiency frontier, throughput vs BE:
[math_eval](../results/passK/L_L1_fig4_efficiency_frontier_math_eval.png) ·
[gsm8k_eval](../results/passK/L_L1_fig4_efficiency_frontier_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/L_L1_fig4_efficiency_frontier_olympiad_eval.png)

- Plotting throughput against BE directly (top-right = better) shows L=16's
  cluster (medium blue) sitting **above and often to the right of** L=32's
  cluster (dark blue) — L=16 with L1 in the 8–12 range reaches comparable or
  better BE than plain L=32, at roughly double the throughput.
- L=32 is not "the slow-but-thorough option" here — for these two verifiers,
  it's dominated. The sweet spot in this data is **L=16, L1≈8–10**, not L=32
  at any L1.

## 4. Same pattern on a second checkpoint (jsd_flat_enrich_K3) — and it gains *more* from L1 than plain jsd

enrich checkpoint, traversal, BE and throughput vs L1:
[math_eval](../results/passK/L_L1_fig5_be_throughput_vs_L1_enrich_traversal_math_eval.png) ·
[gsm8k_eval](../results/passK/L_L1_fig5_be_throughput_vs_L1_enrich_traversal_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/L_L1_fig5_be_throughput_vs_L1_enrich_traversal_olympiad_eval.png)

Not just a qualitative confirmation — enrich's relative gain from L1 is
consistently larger than jsd's, at every (verifier, L) combination checked
(`math_eval` numbers below; same direction on the other two datasets):

| verifier | L | enrich Δ (L1=0 → best available) | jsd Δ (L1=0 → best available) |
|---|---|---|---|
| traversal | 32 | +26.0% (8.36→10.53, L1=8) | +17.1% (8.93→10.45, L1=12) |
| specinfer | 32 | +61.2% (4.97→8.01, L1=9) | +46.9% (5.31→7.80, L1=9) |
| traversal | 16 | +10.5% (7.51→8.30, L1=9) | +12.0% (7.56→8.46, L1=8) |
| specinfer | 16 | +45.1% (5.10→7.40, L1=9) | +37.3% (5.32→7.31, L1=10) |

jsd vs enrich head to head, L=32:
[math_eval](../results/passK/L_L1_fig6_jsd_vs_enrich_L32_math_eval.png) ·
[gsm8k_eval](../results/passK/L_L1_fig6_jsd_vs_enrich_L32_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/L_L1_fig6_jsd_vs_enrich_L32_olympiad_eval.png)

At L=32, enrich starts **behind** jsd at L1=0 (both verifiers) but rises
faster and **overtakes jsd by L1=8–9** — peak BE 10.53 (enrich) vs 10.45
(jsd) for traversal, 8.01 vs 7.80 for specinfer. At L=16 jsd stays slightly
ahead through the whole range tested. Enrich's L=32 sweep is only complete
through L1=9 (still running at time of writing) — the crossover holds in
the data available now, not yet confirmed at L1=10/12.

**Practical read: if training an enrich checkpoint, don't judge its L1
sensitivity from L1=0 alone** — the L1=0 BE understates what enrich can
reach once L1 is tuned, more so than for plain jsd.

## 5. Best (checkpoint, verifier, L, L1) combination found so far

Ranked directly across every real data point in hand (L=16/32 only, L=8
excluded as unreliable):

| optimizing for | winner | BE | throughput |
|---|---|---|---|
| **throughput** | jsd, traversal, L=16, L1=10 | 8.39 | **12.84 tok/s** |
| **BE** | enrich, traversal, L=32, L1=8 | **10.53** | 8.68 tok/s |

These are two different answers depending on what you're optimizing for —
the best-throughput pick trades away about 20% of the best-BE pick's block
efficiency for ~48% more tokens/sec. If wall-clock speed is what matters,
`jsd` at `L=16, L1≈10` is the best combination found so far, not `L=32` at
any L1, and not the `enrich` checkpoint.

## 6. Confirmatory rerun (jsd only, full L1 grid, all 3 datasets)

A second, complete pass over jsd's L=16/32 sweep
([results/passK/LandL1sweepJsd.csv](../results/passK/LandL1sweepJsd.csv), combining
[l_sweep_jsd_rerun.csv](../results/passK/l_sweep_jsd_rerun.csv) (L1=0, all 9
verifiers) and [l1_sweep_jsd_rerun.csv](../results/passK/l1_sweep_jsd_rerun.csv)
(L1=6/8/9/10/12, traversal+specinfer)) — this time with `olympiad_eval` fully
populated at both L=16 and L=32. One chart per dataset, BE (solid, left axis)
and throughput (dashed hue, right axis) together, both verifiers, both L:

[math_eval](../results/passK/LandL1_jsd_report_math_eval.png) ·
[gsm8k_eval](../results/passK/LandL1_jsd_report_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/LandL1_jsd_report_olympiad_eval.png)

- The core finding holds on a fresh run and on all 3 datasets: **BE and
  throughput rise together as L1 increases**, for both `traversal` and
  `specinfer`, at both L=16 and L=32 — no dataset breaks the pattern.
- `specinfer` still gains more than `traversal` in relative terms on every
  dataset (e.g. math_eval L=32: specinfer 5.3→7.8 BE vs traversal 8.9→10.4).
- Small non-monotonic dips appear at individual L1 values (e.g. L1=9 dipping
  slightly on some traversal/L=16 lines) — single-run noise, not a reversal;
  the overall L1=0→12 trend is monotone-up on every line in every chart.

## 7. Confirmatory rerun (jsd_flat_enrich_K3 only, full L1 grid, all 3 datasets)

Same treatment as section 6, run for the enrich checkpoint
([results/passK/LandL1sweepJsdEnrich.csv](../results/passK/LandL1sweepJsdEnrich.csv), combining
[l_sweep_jsd_enrich_final.csv](../results/passK/l_sweep_jsd_enrich_final.csv) (L1=0, all 9
verifiers) and [l1_sweep_jsd_enrich_final.csv](../results/passK/l1_sweep_jsd_enrich_final.csv)
(L1=6/8/9/10/12, traversal+specinfer)) — all 3 datasets complete at L=16/32.

[math_eval](../results/passK/LandL1_jsdenrich_report_math_eval.png) ·
[gsm8k_eval](../results/passK/LandL1_jsdenrich_report_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/LandL1_jsdenrich_report_olympiad_eval.png)

- Same core finding as jsd: BE and throughput rise together as L1 increases,
  for both `traversal` and `specinfer`, at both L=16 and L=32, on all 3
  datasets.
- The one wrinkle enrich shows that jsd's rerun didn't: `traversal` at L=32
  dips noticeably at L1=9–10 before recovering by L1=12, on all 3 datasets
  (e.g. math_eval: 10.53 at L1=8 → 9.83 at L1=10 → 9.99 at L1=12). Still a
  single run — treat as a soft spot in the L1=9–10 region for this
  particular (checkpoint, verifier, L) combination, not a reversal of the
  overall upward trend.
- Confirms section 4's finding that enrich gains more from L1 than jsd in
  relative terms, on a second, independent full rerun.

## Practical read

- Don't tune L1 using L=8 data — the noise floor there swamps the signal.
- If deploying at L=16 or L=32, sweep L1 in the 8–12 range before committing
  to the L1=0 (full branching) default — that default is very likely leaving
  both BE and throughput on the table simultaneously, at least for
  `traversal` and `specinfer`.
- `specinfer` has more to gain from L1 tuning than `traversal` — if choosing
  which verifier to pair with delayed branching, specinfer's headroom is
  larger.
