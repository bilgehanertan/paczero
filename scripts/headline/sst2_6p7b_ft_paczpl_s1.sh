#!/bin/bash
# Table 1, SST-2 6.7B FT PAC-ZPL (seed 1)
# Paper reference: experiments.tex Table 1
# Recipe: lr=1e-4 c=1000 T=1000 M=126 (PAC-ZPL plain, MI=0) seed=1
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl sst2_6p7b_ft_paczpl_s1 facebook/opt-6.7b SST2 ft 1e-4 1000 1000 126 1
