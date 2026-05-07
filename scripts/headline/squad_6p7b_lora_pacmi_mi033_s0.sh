#!/bin/bash
# Table 1, SQuAD 6.7B LoRA PAC-MI MI=0.33 (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-3 c=10 T=1000 MI=0.33 M=128
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi squad_6p7b_lora_pacmi_mi033_s0 facebook/opt-6.7b SQuAD lora 1e-3 10 1000 0.33 128 0
