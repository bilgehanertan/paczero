#!/bin/bash
# DP-AggZO K=64 FT, eps=2.0
# Paper reference: experiments.tex Table dp-cliff + appendix Table dpaggzo-parity
# Recipe: ft lr=5e-6 c=25 sample_rate=0.064 T=1000 N=64 eps=2.0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_dpaggzo dpaggzo_sst2_1p3b_ft_eps2p0_K64_T1000 facebook/opt-1.3b SST2 ft 5e-6 1000 2.0 25 64 0.064 0
