#!/bin/bash
# Table 1, SQuAD 6.7B FT PAC-MI MI=0.68 (single seed)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 MI=0.68 M=128
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi squad_6p7b_ft_pacmi_mi068_s0 facebook/opt-6.7b SQuAD ft 1e-4 1000 1000 0.68 128 0
