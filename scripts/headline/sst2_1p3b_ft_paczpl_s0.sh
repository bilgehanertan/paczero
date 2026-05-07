#!/bin/bash
# Table 1, SST-2 1.3B FT PAC-ZPL (seed 0)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 M=126 (PAC-ZPL plain, MI=0) seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl sst2_1p3b_ft_paczpl_s0 facebook/opt-1.3b SST2 ft 1e-4 1000 1000 126 0
