#!/bin/bash
# Table 1, SQuAD 1.3B LoRA PAC-ZPL (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-3 c=25 T=1000 M=126 (PAC-ZPL plain, MI=0)
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl squad_1p3b_lora_paczpl_s0 facebook/opt-1.3b SQuAD lora 1e-3 25 1000 126 0
