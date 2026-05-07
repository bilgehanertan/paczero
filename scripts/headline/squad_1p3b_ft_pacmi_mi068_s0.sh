#!/bin/bash
# Table 1, SQuAD 1.3B FT PAC-MI MI=0.68 (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 MI=0.68 M=128
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi squad_1p3b_ft_pacmi_mi068_s0 facebook/opt-1.3b SQuAD ft 1e-4 1000 1000 0.68 128 0
