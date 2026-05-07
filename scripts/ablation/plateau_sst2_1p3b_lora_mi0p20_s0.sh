#!/bin/bash
# Tight-MI plateau (LoRA), MI=0.20
# Paper reference: experiments.tex Table 2 (Tight-MI plateau)
# Recipe: lr=5e-4 c=25 T=2000 MI=0.20 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi plateau_sst2_1p3b_lora_mi0p20_s0 facebook/opt-1.3b SST2 lora 5e-4 25 2000 0.20 128 0
