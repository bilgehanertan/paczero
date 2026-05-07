#!/bin/bash
# Table 1, SST-2 6.7B LoRA PAC-MI MI=0.33 (seed 1)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-3 c=10 T=2000 MI=0.33 M=128 seed=1
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi sst2_6p7b_lora_pacmi_mi033_s1 facebook/opt-6.7b SST2 lora 1e-3 10 2000 0.33 128 1
