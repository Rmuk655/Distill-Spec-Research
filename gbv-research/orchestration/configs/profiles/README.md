# Experiment Profiles

Profiles are YAML overrides that narrow what a base config runs.
A profile is a complete, standalone config — copy the relevant base config
(colab_a100.yaml, laptop.yaml, …) and add an `experiment:` section.

## How to use

```bash
# Run with a profile config
python orchestration/experiment.py --config profiles/promote_kl_jsd_l1 --yes

# Or override losses on the fly without a config file
python orchestration/experiment.py --config colab_a100 --losses kl,jsd,l1 --yes
```

The `--config` argument resolves against `orchestration/configs/` automatically,
so `--config profiles/promote_kl_jsd_l1` loads
`orchestration/configs/profiles/promote_kl_jsd_l1.yaml`.

## Available profiles

| Profile | Base | Purpose |
|---------|------|---------|
| `profiles/promote_kl_jsd_l1` | `colab_a100` | A100 paper runs for KL, JSD, L1 only |
| `profiles/online_only_laptop` | `laptop` | Mukund: online (KL) smoke + train + eval only, no offline losses |
| `profiles/online_only_colab_lite` | `colab_lite` | Online trend check on T4 |

## Adding a new profile

1. Copy the nearest base config: `cp colab_a100.yaml profiles/my_ablation.yaml`
2. Edit the `experiment:` section:
   ```yaml
   experiment:
     losses: [kl, ebe]        # run only these losses
     seed_override: 123       # optional: use a different seed
   ```
3. Run: `python orchestration/experiment.py --config profiles/my_ablation --yes`

## `experiment:` section reference

```yaml
experiment:
  losses: [kl, jsd, l1]    # Subset of: kl ebe ebe_single rev_kl jsd l1
                             #            online online_ebe online_ebe_single
                             # Omit or set to null for all losses.
  seed_override: 123         # Override training.seed without editing the base config.
                             # Useful for second-seed reproducibility runs.
```

The `--losses` CLI flag always overrides the YAML `experiment.losses` field.
```bash
# YAML has losses: [kl, jsd, l1] but you want to add l1 quickly:
python experiment.py --config profiles/promote_kl_jsd_l1 --losses kl,jsd,l1,rev_kl --yes
```
