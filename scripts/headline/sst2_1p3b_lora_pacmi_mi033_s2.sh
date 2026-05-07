#!/bin/bash
# Table 1, SST-2 1.3B LoRA PAC-MI MI=0.33 (seed 2)
# Paper reference: experiments.tex Table 1
# Recipe: lr=5e-4 c=25 T=2000 MI=0.33 M=128 seed=2
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi sst2_1p3b_lora_pacmi_mi033_s2 facebook/opt-1.3b SST2 lora 5e-4 25 2000 0.33 128 2
