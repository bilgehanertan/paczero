#!/bin/bash
# Table 1, SQuAD 6.7B FT PAC-ZPL (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 M=126 (PAC-ZPL plain, MI=0)
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl squad_6p7b_ft_paczpl_s0 facebook/opt-6.7b SQuAD ft 1e-4 1000 1000 126 0
