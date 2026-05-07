#!/bin/bash
# Table 1, SST-2 1.3B LoRA PAC-ZPL (seed 1)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-3 c=25 T=1000 M=126 (PAC-ZPL plain, MI=0) seed=1
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl sst2_1p3b_lora_paczpl_s1 facebook/opt-1.3b SST2 lora 1e-3 25 1000 126 1
