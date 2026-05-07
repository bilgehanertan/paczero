#!/bin/bash
# PAC-ZPL canonical 6.7B LoRA T-ladder (post-hoc), seed=2
# Paper reference: appendix.tex Table zpl-67b-ladder
# Recipe: lora lr=1e-3 c=25 T=1500 M=128 seed=2
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl zplladder_sst2_6p7b_lora_c25_T1500_s2 facebook/opt-6.7b SST2 lora 1e-3 25 1500 128 2
