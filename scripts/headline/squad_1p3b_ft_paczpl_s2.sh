#!/bin/bash
# Table 1, SQuAD 1.3B FT PAC-ZPL (multi-seed) (seed 2)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 M=126 (PAC-ZPL plain, MI=0) seed=2
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl squad_1p3b_ft_paczpl_s2 facebook/opt-1.3b SQuAD ft 1e-4 1000 1000 126 2
