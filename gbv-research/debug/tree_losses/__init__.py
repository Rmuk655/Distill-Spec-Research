"""
debug/tree_losses — a tiny, fully-visualisable sandbox for understanding the
tree-structured distillation losses and their matching tree verifiers.

This package is for STUDENT LEARNING and DEBUGGING only. It is never imported
by the training or evaluation pipeline. It deliberately uses toy order-1
Markov "models" over a 5-8 token vocabulary so that every distribution, every
draft tree, and every loss term can be printed and drawn on one screen — while
still exercising the *real* loss code (distillspec_gbv.losses.tree_losses) and
the *real* verifier code (distillspec_gbv.verifiers.otlp_registry) as the
source of truth.

Modules
-------
tiny_models      TinyMarkovModel (teacher/student) + tree sampling in the exact
                 dict format the real losses & verifiers consume.
tree_harness     Monte-Carlo block-efficiency estimator that drives the REAL
                 TreeVerifier, plus fresh-dict / cache-clear plumbing.
loss_tracer      Transparent per-node re-derivation of each loss, validated
                 against the real compute_tree_loss scalar.
visual_debugger  matplotlib/networkx visual stepper + train-and-watch demo (CLI).
"""
