#!/bin/bash
# Table 1, SST-2 1.3B FT PAC-MI MI=0.33 (seed 0)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 MI=0.33 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi sst2_1p3b_ft_pacmi_mi033_s0 facebook/opt-1.3b SST2 ft 1e-4 1000 1000 0.33 128 0
