#!/bin/bash
# Non-private SQuAD reference (6.7B FT, Liu24-canonical)
# Paper reference: appendix.tex Table squad-nonpriv
# Recipe: ft lr=1e-5 c=100 T=1000 M=128 seed=0 no_privacy=True
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_nonpriv nonpriv_squad_6p7b_ft_lr1em5_c100_T1000_s0 facebook/opt-6.7b SQuAD ft 1e-5 100 1000 128 0 none
