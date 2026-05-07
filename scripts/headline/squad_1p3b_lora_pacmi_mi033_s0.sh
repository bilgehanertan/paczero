#!/bin/bash
# Table 1, SQuAD 1.3B LoRA PAC-MI MI=0.33 (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-3 c=25 T=1000 MI=0.33 M=128
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi squad_1p3b_lora_pacmi_mi033_s0 facebook/opt-1.3b SQuAD lora 1e-3 25 1000 0.33 128 0
