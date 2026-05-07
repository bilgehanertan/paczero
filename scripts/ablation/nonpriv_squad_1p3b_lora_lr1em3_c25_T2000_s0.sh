#!/bin/bash
# Non-private SQuAD reference (1.3B LoRA)
# Paper reference: appendix.tex Table squad-nonpriv
# Recipe: lora lr=1e-3 c=25 T=2000 M=128 seed=0 no_privacy=True
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_nonpriv nonpriv_squad_1p3b_lora_lr1em3_c25_T2000_s0 facebook/opt-1.3b SQuAD lora 1e-3 25 2000 128 0 none
