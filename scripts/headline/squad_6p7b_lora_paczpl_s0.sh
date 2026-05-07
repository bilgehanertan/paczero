#!/bin/bash
# Table 1, SQuAD 6.7B LoRA PAC-ZPL (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-3 c=10 T=1000 M=126 (PAC-ZPL plain, MI=0)
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl squad_6p7b_lora_paczpl_s0 facebook/opt-6.7b SQuAD lora 1e-3 10 1000 126 0
