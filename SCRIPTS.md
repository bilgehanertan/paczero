# SCRIPTS.md — script ↔ paper cell map

Every script under `scripts/` reproduces one cell reported in the paper.
Each row below names the paper table the cell appears in, what the cell does, and the dev-tuned recipe (lr / clip / T / M / MI / K / seed)
the script invokes. Expected results (test/dev metrics) are reported in the
paper itself — see the table referenced in the row.

**Smoke mode**: prepend `SMOKE=1` to any script for a 1–2 minute sanity check.
Smoke overrides `--max_steps` (1), `--num_train` (128), `--num_dev` (10),
`--num_eval` (20), `--per_device_train_batch_size` (128) and disables the
dev-best save pipeline so the trainer just verifies the script launches and
metrics.json is written. Each smoke cell finishes in ≲30 s on H100 80 GB.

## Headline Table 1 — SST-2 PAC-MI cells (n≥3 multi-seed)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/headline/sst2_1p3b_lora_pacmi_mi033_s0.sh` | Table 1, row LoRA · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.33 nats, seed 0 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi033_s1.sh` | Table 1, row LoRA · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.33 nats, seed 1 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi033_s2.sh` | Table 1, row LoRA · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.33 nats, seed 2 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi033_s3.sh` | Table 1, row LoRA · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.33 nats, seed 3 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi068_s0.sh` | Table 1, row LoRA · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.68 nats, seed 0 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi068_s1.sh` | Table 1, row LoRA · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.68 nats, seed 1 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi068_s2.sh` | Table 1, row LoRA · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.68 nats, seed 2 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_lora_pacmi_mi068_s3.sh` | Table 1, row LoRA · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B and LoRA r=8 at MI budget 0.68 nats, seed 3 | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_lora_pacmi_mi033_s0.sh` | Table 1, row LoRA · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B and LoRA r=8 at MI=0.33, seed 0 | lr=1e-3, clip c=10, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_lora_pacmi_mi033_s1.sh` | Table 1, row LoRA · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B and LoRA r=8 at MI=0.33, seed 1 | lr=1e-3, clip c=10, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_lora_pacmi_mi033_s2.sh` | Table 1, row LoRA · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B and LoRA r=8 at MI=0.33, seed 2 | lr=1e-3, clip c=10, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_lora_pacmi_mi068_s0.sh` | Table 1, row LoRA · MI=0.68 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B and LoRA r=8 at MI=0.68, seed 0 | lr=1e-3, clip c=10, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_lora_pacmi_mi068_s1.sh` | Table 1, row LoRA · MI=0.68 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B and LoRA r=8 at MI=0.68, seed 1 | lr=1e-3, clip c=10, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_lora_pacmi_mi068_s2.sh` | Table 1, row LoRA · MI=0.68 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B and LoRA r=8 at MI=0.68, seed 2 | lr=1e-3, clip c=10, T=2000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi033_s0.sh` | Table 1, row FT · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.33, seed 0 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi033_s1.sh` | Table 1, row FT · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.33, seed 1 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi033_s2.sh` | Table 1, row FT · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.33, seed 2 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi033_s3.sh` | Table 1, row FT · MI=0.33 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.33, seed 3 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi068_s0.sh` | Table 1, row FT · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.68, seed 0 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi068_s1.sh` | Table 1, row FT · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.68, seed 1 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi068_s2.sh` | Table 1, row FT · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.68, seed 2 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_1p3b_ft_pacmi_mi068_s3.sh` | Table 1, row FT · MI=0.68 · SST-2 1.3B | PAC-MI on SST-2 with OPT-1.3B full-parameter FT at MI=0.68, seed 3 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi033_s0.sh` | Table 1, row FT · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.33, seed 0 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi033_s1.sh` | Table 1, row FT · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.33, seed 1 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi033_s2.sh` | Table 1, row FT · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.33, seed 2 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi033_s42.sh` | Table 1, row FT · MI=0.33 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.33, seed 42 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi068_s0.sh` | Table 1, row FT · MI=0.68 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.68, seed 0 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi068_s1.sh` | Table 1, row FT · MI=0.68 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.68, seed 1 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/sst2_6p7b_ft_pacmi_mi068_s2.sh` | Table 1, row FT · MI=0.68 · SST-2 6.7B | PAC-MI on SST-2 with OPT-6.7B full-parameter FT at MI=0.68, seed 2 | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |

## Headline Table 1 — SST-2 PAC-ZPL cells (n=3 multi-seed)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/headline/sst2_1p3b_lora_paczpl_s1.sh` | Table 1, row LoRA · MI≡0 · SST-2 1.3B (PAC-ZPL) | PAC-ZPL (zero MI) on SST-2 with OPT-1.3B and LoRA r=8, seed 1 | lr=1e-3, clip c=25, T=1000, M=126 (multi-seed pool at headline recipe) |
| `scripts/headline/sst2_1p3b_lora_paczpl_s2.sh` | Table 1, row LoRA · MI≡0 · SST-2 1.3B (PAC-ZPL) | PAC-ZPL (zero MI) on SST-2 with OPT-1.3B and LoRA r=8, seed 2 | lr=1e-3, clip c=25, T=1000, M=126 (multi-seed pool at headline recipe) |
| `scripts/headline/sst2_1p3b_lora_paczpl_s3.sh` | Table 1, row LoRA · MI≡0 · SST-2 1.3B (PAC-ZPL) | PAC-ZPL (zero MI) on SST-2 with OPT-1.3B and LoRA r=8, seed 3 | lr=1e-3, clip c=25, T=1000, M=126 (multi-seed pool at headline recipe) |
| `scripts/headline/sst2_1p3b_ft_paczpl_s0.sh` | Table 1, row FT · MI≡0 · SST-2 1.3B (PAC-ZPL) | PAC-ZPL (zero MI) on SST-2 with OPT-1.3B full-parameter FT, seed 0 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/sst2_1p3b_ft_paczpl_s1.sh` | Table 1, row FT · MI≡0 · SST-2 1.3B (PAC-ZPL) | PAC-ZPL (zero MI) on SST-2 with OPT-1.3B full-parameter FT, seed 1 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/sst2_1p3b_ft_paczpl_s2.sh` | Table 1, row FT · MI≡0 · SST-2 1.3B (PAC-ZPL) | PAC-ZPL (zero MI) on SST-2 with OPT-1.3B full-parameter FT, seed 2 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/sst2_6p7b_lora_paczpl_s0.sh` | Table 1, row LoRA · MI≡0 · SST-2 6.7B (PAC-ZPL) | PAC-ZPL on SST-2 with OPT-6.7B and LoRA r=8, seed 0 | lr=1e-3, clip c=10, T=1000, M=126 |
| `scripts/headline/sst2_6p7b_lora_paczpl_s1.sh` | Table 1, row LoRA · MI≡0 · SST-2 6.7B (PAC-ZPL) | PAC-ZPL on SST-2 with OPT-6.7B and LoRA r=8, seed 1 | lr=1e-3, clip c=10, T=1000, M=126 |
| `scripts/headline/sst2_6p7b_lora_paczpl_s2.sh` | Table 1, row LoRA · MI≡0 · SST-2 6.7B (PAC-ZPL) | PAC-ZPL on SST-2 with OPT-6.7B and LoRA r=8, seed 2 | lr=1e-3, clip c=10, T=1000, M=126 |
| `scripts/headline/sst2_6p7b_ft_paczpl_s0.sh` | Table 1, row FT · MI≡0 · SST-2 6.7B (PAC-ZPL) | PAC-ZPL on SST-2 with OPT-6.7B full-parameter FT, seed 0 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/sst2_6p7b_ft_paczpl_s1.sh` | Table 1, row FT · MI≡0 · SST-2 6.7B (PAC-ZPL) | PAC-ZPL on SST-2 with OPT-6.7B full-parameter FT, seed 1 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/sst2_6p7b_ft_paczpl_s2.sh` | Table 1, row FT · MI≡0 · SST-2 6.7B (PAC-ZPL) | PAC-ZPL on SST-2 with OPT-6.7B full-parameter FT, seed 2 | lr=1e-4, clip c=1000, T=1000, M=126 |

## Headline Table 1 — SQuAD PAC-MI cells (single-seed; PAC-ZPL multi-seed for 1.3B FT)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/headline/squad_1p3b_lora_pacmi_mi033_s0.sh` | Table 1, row LoRA · MI=0.33 · SQuAD 1.3B | PAC-MI on SQuAD with OPT-1.3B + LoRA r=8 at MI=0.33 (single seed) | lr=1e-3, clip c=25, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_1p3b_lora_pacmi_mi068_s0.sh` | Table 1, row LoRA · MI=0.68 · SQuAD 1.3B | PAC-MI on SQuAD with OPT-1.3B + LoRA r=8 at MI=0.68 (single seed) | lr=1e-3, clip c=25, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_6p7b_lora_pacmi_mi033_s0.sh` | Table 1, row LoRA · MI=0.33 · SQuAD 6.7B | PAC-MI on SQuAD with OPT-6.7B + LoRA r=8 at MI=0.33 (single seed) | lr=1e-3, clip c=10, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_6p7b_lora_pacmi_mi068_s0.sh` | Table 1, row LoRA · MI=0.68 · SQuAD 6.7B | PAC-MI on SQuAD with OPT-6.7B + LoRA r=8 at MI=0.68 (single seed) | lr=1e-3, clip c=10, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_1p3b_ft_pacmi_mi033_s0.sh` | Table 1, row FT · MI=0.33 · SQuAD 1.3B | PAC-MI on SQuAD with OPT-1.3B full-parameter FT at MI=0.33 (single seed) | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_1p3b_ft_pacmi_mi068_s0.sh` | Table 1, row FT · MI=0.68 · SQuAD 1.3B | PAC-MI on SQuAD with OPT-1.3B full-parameter FT at MI=0.68 (single seed) | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_6p7b_ft_pacmi_mi033_s0.sh` | Table 1, row FT · MI=0.33 · SQuAD 6.7B | PAC-MI on SQuAD with OPT-6.7B full-parameter FT at MI=0.33 (single seed) | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_6p7b_ft_pacmi_mi068_s0.sh` | Table 1, row FT · MI=0.68 · SQuAD 6.7B | PAC-MI on SQuAD with OPT-6.7B full-parameter FT at MI=0.68 (single seed) | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t |
| `scripts/headline/squad_1p3b_lora_paczpl_s0.sh` | Table 1, row LoRA · MI≡0 · SQuAD 1.3B (PAC-ZPL) | PAC-ZPL on SQuAD with OPT-1.3B + LoRA r=8 (single seed) | lr=1e-3, clip c=25, T=1000, M=126 |
| `scripts/headline/squad_6p7b_lora_paczpl_s0.sh` | Table 1, row LoRA · MI≡0 · SQuAD 6.7B (PAC-ZPL) | PAC-ZPL on SQuAD with OPT-6.7B + LoRA r=8 (single seed) | lr=1e-3, clip c=10, T=1000, M=126 |
| `scripts/headline/squad_1p3b_ft_paczpl_s0.sh` | Table 1, row FT · MI≡0 · SQuAD 1.3B (PAC-ZPL multi-seed) | PAC-ZPL on SQuAD with OPT-1.3B full-parameter FT, seed 0 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/squad_1p3b_ft_paczpl_s1.sh` | Table 1, row FT · MI≡0 · SQuAD 1.3B (PAC-ZPL multi-seed) | PAC-ZPL on SQuAD with OPT-1.3B full-parameter FT, seed 1 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/squad_1p3b_ft_paczpl_s2.sh` | Table 1, row FT · MI≡0 · SQuAD 1.3B (PAC-ZPL multi-seed) | PAC-ZPL on SQuAD with OPT-1.3B full-parameter FT, seed 2 | lr=1e-4, clip c=1000, T=1000, M=126 |
| `scripts/headline/squad_6p7b_ft_paczpl_s0.sh` | Table 1, row FT · MI≡0 · SQuAD 6.7B (PAC-ZPL) | PAC-ZPL on SQuAD with OPT-6.7B full-parameter FT (single seed) | lr=1e-4, clip c=1000, T=1000, M=126 |

## Tight-MI plateau (Table 2 + Appendix FT-plateau)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/plateau_sst2_1p3b_lora_mi1em4_s0.sh` | Table 2 (LoRA plateau), MI=1e-4 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=1e-4 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi3em4_s0.sh` | Table 2 (LoRA plateau), MI=3e-4 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=3e-4 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi5em4_s0.sh` | Table 2 (LoRA plateau), MI=5e-4 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=5e-4 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi1em3_s0.sh` | Table 2 (LoRA plateau), MI=1e-3 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=1e-3 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi2em3_s0.sh` | Table 2 (LoRA plateau), MI=2e-3 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=2e-3 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi3em3_s0.sh` | Table 2 (LoRA plateau), MI=3e-3 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=3e-3 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi1em2_s0.sh` | Table 2 (LoRA plateau), MI=1e-2 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=1e-2 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi3em2_s0.sh` | Table 2 (LoRA plateau), MI=3e-2 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=3e-2 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi5em2_s0.sh` | Table 2 (LoRA plateau), MI=5e-2 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=5e-2 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi0p07_s0.sh` | Table 2 (LoRA plateau), MI=0.07 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=0.07 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi0p11_s0.sh` | Table 2 (LoRA plateau), MI=0.11 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=0.11 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi0p20_s0.sh` | Table 2 (LoRA plateau), MI=0.20 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=0.20 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi0p33_s0.sh` | Table 2 (LoRA plateau), MI=0.33 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=0.33 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi0p50_s0.sh` | Table 2 (LoRA plateau), MI=0.50 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=0.50 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_lora_mi0p68_s0.sh` | Table 2 (LoRA plateau), MI=0.68 | PAC-MI tight-MI plateau (LoRA): SST-2 OPT-1.3B at MI=0.68 nats | lr=5e-4, clip c=25, T=2000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi1em4_s0.sh` | Appendix Table FT-plateau-adaptive, MI=1e-4 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=1e-4 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi3em4_s0.sh` | Appendix Table FT-plateau-adaptive, MI=3e-4 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=3e-4 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi5em4_s0.sh` | Appendix Table FT-plateau-adaptive, MI=5e-4 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=5e-4 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi1em3_s0.sh` | Appendix Table FT-plateau-adaptive, MI=1e-3 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=1e-3 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi2em3_s0.sh` | Appendix Table FT-plateau-adaptive, MI=2e-3 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=2e-3 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi3em3_s0.sh` | Appendix Table FT-plateau-adaptive, MI=3e-3 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=3e-3 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi1em2_s0.sh` | Appendix Table FT-plateau-adaptive, MI=1e-2 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=1e-2 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi3em2_s0.sh` | Appendix Table FT-plateau-adaptive, MI=3e-2 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=3e-2 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi5em2_s0.sh` | Appendix Table FT-plateau-adaptive, MI=5e-2 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=5e-2 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi0p07_s0.sh` | Appendix Table FT-plateau-adaptive, MI=0.07 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=0.07 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi0p11_s0.sh` | Appendix Table FT-plateau-adaptive, MI=0.11 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=0.11 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi0p20_s0.sh` | Appendix Table FT-plateau-adaptive, MI=0.20 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=0.20 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi0p33_s0.sh` | Appendix Table FT-plateau-adaptive, MI=0.33 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=0.33 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi0p50_s0.sh` | Appendix Table FT-plateau-adaptive, MI=0.50 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=0.50 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |
| `scripts/ablation/plateau_sst2_1p3b_ft_mi0p68_s0.sh` | Appendix Table FT-plateau-adaptive, MI=0.68 | PAC-MI FT plateau: SST-2 OPT-1.3B at MI=0.68 nats | lr=1e-4, clip c=1000, T=1000, M=128, adaptive β_t, seed 0 |

## 6.7B LoRA clip ablation (Appendix Table clip-ablation-67b)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/clipabl_sst2_6p7b_lora_c10_mi0p33_s0.sh` | Appendix Table clip-ablation-67b, c=10 · MI=0.33 | PAC-MI 6.7B LoRA clip ablation: clip c=10 at MI=0.33 | lr=1e-3, clip c=10, T=1000, M=128, MI=0.33, adaptive β_t, seed 0 |
| `scripts/ablation/clipabl_sst2_6p7b_lora_c25_mi0p33_s0.sh` | Appendix Table clip-ablation-67b, c=25 · MI=0.33 | PAC-MI 6.7B LoRA clip ablation: clip c=25 at MI=0.33 | lr=1e-3, clip c=25, T=1000, M=128, MI=0.33, adaptive β_t, seed 0 |
| `scripts/ablation/clipabl_sst2_6p7b_lora_c50_mi0p33_s0.sh` | Appendix Table clip-ablation-67b, c=50 · MI=0.33 | PAC-MI 6.7B LoRA clip ablation: clip c=50 at MI=0.33 | lr=1e-3, clip c=50, T=1000, M=128, MI=0.33, adaptive β_t, seed 0 |
| `scripts/ablation/clipabl_sst2_6p7b_lora_c10_mi0p68_s0.sh` | Appendix Table clip-ablation-67b, c=10 · MI=0.68 | PAC-MI 6.7B LoRA clip ablation: clip c=10 at MI=0.68 | lr=1e-3, clip c=10, T=1000, M=128, MI=0.68, adaptive β_t, seed 0 |

## LoRA rank ablation (Appendix Table rank-ablation)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/rankabl_sst2_6p7b_lora_mi0p33_r4_s0.sh` | Appendix Table rank-ablation, r=4 | PAC-MI 6.7B LoRA rank ablation: rank r=4 at MI=0.33 c=25 | lr=1e-3, clip c=25, T=1000, M=128, r=4, MI=0.33, adaptive β_t, seed 0 |
| `scripts/ablation/rankabl_sst2_6p7b_lora_mi0p33_r8_s0.sh` | Appendix Table rank-ablation, r=8 | PAC-MI 6.7B LoRA rank ablation: rank r=8 at MI=0.33 c=25 | lr=1e-3, clip c=25, T=1000, M=128, r=8, MI=0.33, adaptive β_t, seed 0 |
| `scripts/ablation/rankabl_sst2_6p7b_lora_mi0p33_r16_s0.sh` | Appendix Table rank-ablation, r=16 | PAC-MI 6.7B LoRA rank ablation: rank r=16 at MI=0.33 c=25 | lr=1e-3, clip c=25, T=1000, M=128, r=16, MI=0.33, adaptive β_t, seed 0 |

## PAC-ZPL LR sweep (Appendix Table zpl-lr-sweep)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_lora_c25_lr1em4_s0.sh` | Appendix Table zpl-lr-sweep (lora, clip=25, lr=1e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B lora: clip=25, lr=1e-4 | lr=1e-4, clip c=25, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_lora_c25_lr5em4_s0.sh` | Appendix Table zpl-lr-sweep (lora, clip=25, lr=5e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B lora: clip=25, lr=5e-4 | lr=5e-4, clip c=25, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_lora_c25_lr1em3_s0.sh` | Appendix Table zpl-lr-sweep (lora, clip=25, lr=1e-3) | PAC-ZPL LR sweep on SST-2 OPT-1.3B lora: clip=25, lr=1e-3 | lr=1e-3, clip c=25, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_lora_c25_lr2em3_s0.sh` | Appendix Table zpl-lr-sweep (lora, clip=25, lr=2e-3) | PAC-ZPL LR sweep on SST-2 OPT-1.3B lora: clip=25, lr=2e-3 | lr=2e-3, clip c=25, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_lora_noclip_lr2em4_s0.sh` | Appendix Table zpl-lr-sweep (lora, clip=1e9, lr=2e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B lora: clip=1e9, lr=2e-4 | lr=2e-4, clip c=1e9, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_lora_noclip_lr5em4_s0.sh` | Appendix Table zpl-lr-sweep (lora, clip=1e9, lr=5e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B lora: clip=1e9, lr=5e-4 | lr=5e-4, clip c=1e9, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_ft_c1000_lr5em5_s0.sh` | Appendix Table zpl-lr-sweep (ft, clip=1000, lr=5e-5) | PAC-ZPL LR sweep on SST-2 OPT-1.3B ft: clip=1000, lr=5e-5 | lr=5e-5, clip c=1000, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_ft_c1000_lr1em4_s0.sh` | Appendix Table zpl-lr-sweep (ft, clip=1000, lr=1e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B ft: clip=1000, lr=1e-4 | lr=1e-4, clip c=1000, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_ft_c1000_lr5em4_s0.sh` | Appendix Table zpl-lr-sweep (ft, clip=1000, lr=5e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B ft: clip=1000, lr=5e-4 | lr=5e-4, clip c=1000, T=1000, M=128 (sweep cell), seed 0 |
| `scripts/ablation/zpl_lrsweep_sst2_1p3b_ft_noclip_lr1em4_s0.sh` | Appendix Table zpl-lr-sweep (ft, clip=1e9, lr=1e-4) | PAC-ZPL LR sweep on SST-2 OPT-1.3B ft: clip=1e9, lr=1e-4 | lr=1e-4, clip c=1e9, T=1000, M=128 (sweep cell), seed 0 |

## Mechanism decomposition (Appendix Table ablation)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/mech_sst2_1p3b_lora_raw_full_T2000_s0.sh` | Appendix Table ablation, raw_full @ T=2000 | Mechanism decomposition (no_privacy + ablation=raw_full): raw full-batch mean (MeZO baseline) | lr=5e-4, clip c=25, T=2000, M=128, --no_privacy, ablation=raw_full, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_quant_full_T2000_s0.sh` | Appendix Table ablation, quant_full @ T=2000 | Mechanism decomposition (no_privacy + ablation=quant_full): sign of full-batch mean (sign-quant only) | lr=5e-4, clip c=25, T=2000, M=128, --no_privacy, ablation=quant_full, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_quant_full_T1000_s0.sh` | Appendix Table ablation, quant_full @ T=1000 | Mechanism decomposition (no_privacy + ablation=quant_full): sign of full-batch mean at T=1000 | lr=5e-4, clip c=25, T=1000, M=128, --no_privacy, ablation=quant_full, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_raw_half_T2000_s0.sh` | Appendix Table ablation, raw_half @ T=2000 | Mechanism decomposition (no_privacy + ablation=raw_half): secret-subset mean (no quant) | lr=5e-4, clip c=25, T=2000, M=128, --no_privacy, ablation=raw_half, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_raw_half_T1000_s0.sh` | Appendix Table ablation, raw_half @ T=1000 | Mechanism decomposition (no_privacy + ablation=raw_half): secret-subset mean at T=1000 | lr=5e-4, clip c=25, T=1000, M=128, --no_privacy, ablation=raw_half, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_quant_half_T2000_s0.sh` | Appendix Table ablation, quant_half @ T=2000 | Mechanism decomposition (no_privacy + ablation=quant_half): sign(secret-subset mean) at T=2000 | lr=5e-4, clip c=25, T=2000, M=128, --no_privacy, ablation=quant_half, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_quant_half_T1000_s0.sh` | Appendix Table ablation, quant_half @ T=1000 | Mechanism decomposition (no_privacy + ablation=quant_half): sign(secret-subset mean) at T=1000 | lr=5e-4, clip c=25, T=1000, M=128, --no_privacy, ablation=quant_half, seed 0 |
| `scripts/ablation/mech_sst2_1p3b_lora_random_sign_T2000_s0.sh` | Appendix Table ablation, random_sign @ T=2000 | Mechanism decomposition (no_privacy + ablation=random_sign): uncorrelated ±1 (negative control) | lr=5e-4, clip c=25, T=2000, M=128, --no_privacy, ablation=random_sign, seed 0 |

## Canonical PAC-ZPL T-ladders (Appendix zpl-ft-ladder/zpl-67b-ladder/zpl-67b-ft-ladder)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/zplladder_sst2_1p3b_ft_noclip_T1000_s0.sh` | Appendix Table zpl-ft-ladder, T=1000 | Canonical PAC-ZPL FT 1.3B T-ladder rung at T=1000 (no-clip) | lr=1e-4, clip c=1e9 (no-clip), T=1000, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_1p3b_ft_noclip_T2000_s0.sh` | Appendix Table zpl-ft-ladder, T=2000 | Canonical PAC-ZPL FT 1.3B T-ladder rung at T=2000 (no-clip) | lr=1e-4, clip c=1e9 (no-clip), T=2000, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_1p3b_ft_noclip_T3000_s0.sh` | Appendix Table zpl-ft-ladder, T=3000 | Canonical PAC-ZPL FT 1.3B T-ladder rung at T=3000 (no-clip) | lr=1e-4, clip c=1e9 (no-clip), T=3000, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_1p3b_ft_noclip_T4000_s0.sh` | Appendix Table zpl-ft-ladder, T=4000 | Canonical PAC-ZPL FT 1.3B T-ladder rung at T=4000 (no-clip) | lr=1e-4, clip c=1e9 (no-clip), T=4000, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_1p3b_ft_noclip_T5000_s0.sh` | Appendix Table zpl-ft-ladder, T=5000 | Canonical PAC-ZPL FT 1.3B T-ladder rung at T=5000 (no-clip) | lr=1e-4, clip c=1e9 (no-clip), T=5000, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_6p7b_lora_c25_T1500_s0.sh` | Appendix Table zpl-67b-ladder, seed=0 | Canonical PAC-ZPL 6.7B LoRA T-ladder trajectory (T=1500, post-hoc rung), seed 0 | lr=1e-3, clip c=25, T=1500, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_6p7b_lora_c25_T1500_s1.sh` | Appendix Table zpl-67b-ladder, seed=1 | Canonical PAC-ZPL 6.7B LoRA T-ladder trajectory (T=1500, post-hoc rung), seed 1 | lr=1e-3, clip c=25, T=1500, M=128, --pac_zpl, seed 1 |
| `scripts/ablation/zplladder_sst2_6p7b_lora_c25_T1500_s2.sh` | Appendix Table zpl-67b-ladder, seed=2 | Canonical PAC-ZPL 6.7B LoRA T-ladder trajectory (T=1500, post-hoc rung), seed 2 | lr=1e-3, clip c=25, T=1500, M=128, --pac_zpl, seed 2 |
| `scripts/ablation/zplladder_sst2_6p7b_ft_noclip_T1500_s0.sh` | Appendix Tables zpl-67b-ft-ladder + zpl-67b-ft-multiseed, seed=0 | Canonical PAC-ZPL 6.7B FT T-ladder (no-clip) at T=1500, seed 0 | lr=1e-4, clip c=1e9 (no-clip), T=1500, M=128, --pac_zpl, seed 0 |
| `scripts/ablation/zplladder_sst2_6p7b_ft_noclip_T1500_s1.sh` | Appendix Tables zpl-67b-ft-ladder + zpl-67b-ft-multiseed, seed=1 | Canonical PAC-ZPL 6.7B FT T-ladder (no-clip) at T=1500, seed 1 | lr=1e-4, clip c=1e9 (no-clip), T=1500, M=128, --pac_zpl, seed 1 |
| `scripts/ablation/zplladder_sst2_6p7b_ft_noclip_T1500_s2.sh` | Appendix Tables zpl-67b-ft-ladder + zpl-67b-ft-multiseed, seed=2 | Canonical PAC-ZPL 6.7B FT T-ladder (no-clip) at T=1500, seed 2 | lr=1e-4, clip c=1e9 (no-clip), T=1500, M=128, --pac_zpl, seed 2 |

## K-aggregation ablation (Appendix Tables k-ablation + k-ablation-67b)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/kabl_sst2_1p3b_lora_K4_s0.sh` | Appendix Table k-ablation (1.3B LoRA), K=4 | K-aggregation ablation on SST-2 OPT-1.3B LORA: K=4 | lr=5e-4, clip c=25, T=2000, M=128, MI=0.33, K=4, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_1p3b_lora_K16_s0.sh` | Appendix Table k-ablation (1.3B LoRA), K=16 | K-aggregation ablation on SST-2 OPT-1.3B LORA: K=16 | lr=5e-4, clip c=25, T=2000, M=128, MI=0.33, K=16, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_1p3b_ft_K4_s0.sh` | Appendix Table k-ablation (1.3B FT), K=4 | K-aggregation ablation on SST-2 OPT-1.3B FT: K=4 | lr=1e-4, clip c=1000, T=1000, M=128, MI=0.33, K=4, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_1p3b_ft_K16_s0.sh` | Appendix Table k-ablation (1.3B FT), K=16 | K-aggregation ablation on SST-2 OPT-1.3B FT: K=16 | lr=1e-4, clip c=1000, T=1000, M=128, MI=0.33, K=16, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_6p7b_lora_K4_s0.sh` | Appendix Table k-ablation-67b (6.7B LoRA), K=4 | K-aggregation ablation on SST-2 OPT-6.7B LORA: K=4 | lr=1e-3, clip c=10, T=1000, M=128, MI=0.33, K=4, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_6p7b_lora_K16_s0.sh` | Appendix Table k-ablation-67b (6.7B LoRA), K=16 | K-aggregation ablation on SST-2 OPT-6.7B LORA: K=16 | lr=1e-3, clip c=10, T=1000, M=128, MI=0.33, K=16, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_6p7b_ft_K4_s0.sh` | Appendix Table k-ablation-67b (6.7B FT), K=4 | K-aggregation ablation on SST-2 OPT-6.7B FT: K=4 | lr=1e-4, clip c=1000, T=1000, M=128, MI=0.33, K=4, adaptive β_t, seed 0 |
| `scripts/ablation/kabl_sst2_6p7b_ft_K16_s0.sh` | Appendix Table k-ablation-67b (6.7B FT), K=16 | K-aggregation ablation on SST-2 OPT-6.7B FT: K=16 | lr=1e-4, clip c=1000, T=1000, M=128, MI=0.33, K=16, adaptive β_t, seed 0 |

## Non-private SQuAD references (Appendix Table squad-nonpriv)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/ablation/nonpriv_squad_1p3b_lora_lr1em3_c25_T2000_s0.sh` | Appendix Table squad-nonpriv, 1.3B LoRA non-private | Non-private SQuAD reference (1.3B LoRA): dev-tuned recipe | lr=1e-3, clip c=25, T=2000, M=128, --no_privacy, --pac_load_best_dev, seed 0 |
| `scripts/ablation/nonpriv_squad_6p7b_ft_lr1em5_c100_T1000_s0.sh` | Appendix Table squad-nonpriv, 6.7B FT non-private (Liu24-canonical) | Non-private SQuAD reference (6.7B FT): Liu24-canonical recipe | lr=1e-5, clip c=100, T=1000, M=128, --no_privacy, --pac_load_best_dev, seed 0 |

## In-house DP-cliff baselines (Table 3 + Appendix dpaggzo-parity)

| Script | Paper reference | What it does | Recipe |
|---|---|---|---|
| `scripts/baselines/dpzero_sst2_1p3b_ft_eps0p2_K1_T20k.sh` | Table 3, DPZero K=1 FT @ ε=0.2 | In-house DPZero K=1 reproduction (Liu24 recipe): ε=0.2, T=20,000 | lr=5e-6, clip c=25, sample-rate=0.064, T=20,000, K=1, ε=0.2 |
| `scripts/baselines/dpzero_sst2_1p3b_ft_eps0p3_K1_T20k.sh` | Table 3, DPZero K=1 FT @ ε=0.3 | In-house DPZero K=1 reproduction (Liu24 recipe): ε=0.3, T=20,000 | lr=5e-6, clip c=25, sample-rate=0.064, T=20,000, K=1, ε=0.3 |
| `scripts/baselines/dpzero_sst2_1p3b_ft_eps0p5_K1_T20k.sh` | Table 3, DPZero K=1 FT @ ε=0.5 | In-house DPZero K=1 reproduction (Liu24 recipe): ε=0.5, T=20,000 | lr=5e-6, clip c=25, sample-rate=0.064, T=20,000, K=1, ε=0.5 |
| `scripts/baselines/dpzero_sst2_1p3b_ft_eps1p0_K1_T20k.sh` | Table 3, DPZero K=1 FT @ ε=1.0 | In-house DPZero K=1 reproduction (Liu24 recipe): ε=1.0, T=20,000 | lr=5e-6, clip c=25, sample-rate=0.064, T=20,000, K=1, ε=1.0 |
| `scripts/baselines/dpzero_sst2_1p3b_ft_eps2p0_K1_T20k.sh` | Table 3, DPZero K=1 FT @ ε=2.0 | In-house DPZero K=1 reproduction (Liu24 recipe): ε=2.0, T=20,000 | lr=5e-6, clip c=25, sample-rate=0.064, T=20,000, K=1, ε=2.0 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps0p2_K64_T1000.sh` | Table 3 + Appendix dpaggzo-parity, DP-AggZO K=64 FT @ ε=0.2 | In-house DP-AggZO K=64 reproduction (Bao24 recipe): ε=0.2, T=1000 | lr=5e-6, clip c=25, sample-rate=0.064, T=1000, K=64, ε=0.2 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps0p3_K64_T1000.sh` | Table 3 + Appendix dpaggzo-parity, DP-AggZO K=64 FT @ ε=0.3 | In-house DP-AggZO K=64 reproduction (Bao24 recipe): ε=0.3, T=1000 | lr=5e-6, clip c=25, sample-rate=0.064, T=1000, K=64, ε=0.3 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps0p5_K64_T1000.sh` | Table 3 + Appendix dpaggzo-parity, DP-AggZO K=64 FT @ ε=0.5 | In-house DP-AggZO K=64 reproduction (Bao24 recipe): ε=0.5, T=1000 | lr=5e-6, clip c=25, sample-rate=0.064, T=1000, K=64, ε=0.5 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps1p0_K64_T1000.sh` | Table 3 + Appendix dpaggzo-parity, DP-AggZO K=64 FT @ ε=1.0 | In-house DP-AggZO K=64 reproduction (Bao24 recipe): ε=1.0, T=1000 | lr=5e-6, clip c=25, sample-rate=0.064, T=1000, K=64, ε=1.0 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps2p0_K64_T1000.sh` | Table 3 + Appendix dpaggzo-parity, DP-AggZO K=64 FT @ ε=2.0 | In-house DP-AggZO K=64 reproduction (Bao24 recipe): ε=2.0, T=1000 | lr=5e-6, clip c=25, sample-rate=0.064, T=1000, K=64, ε=2.0 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps6p0_K64_T1000.sh` | Table 3 + Appendix dpaggzo-parity, DP-AggZO K=64 FT @ ε=6.0 | In-house DP-AggZO K=64 reproduction (Bao24 recipe): ε=6.0, T=1000 | lr=5e-6, clip c=25, sample-rate=0.064, T=1000, K=64, ε=6.0 |
| `scripts/baselines/dpaggzo_sst2_1p3b_ft_eps0p2_K64_T2000.sh` | Section §dp-cliff (T=2000 confirmation), DP-AggZO K=64 FT @ ε=0.2 | In-house DP-AggZO K=64 at doubled step budget (cliff confirmation) | lr=5e-6, clip c=25, sample-rate=0.064, T=2000, K=64, ε=0.2 |

---