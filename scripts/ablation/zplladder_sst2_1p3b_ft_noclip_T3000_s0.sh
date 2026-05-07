#!/bin/bash
# PAC-ZPL canonical FT T-ladder T=3000
# Paper reference: appendix.tex Table zpl-ft-ladder
# Recipe: ft no-clip lr=1e-4 c=1e9 T=3000 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl zplladder_sst2_1p3b_ft_noclip_T3000_s0 facebook/opt-1.3b SST2 ft 1e-4 1e9 3000 128 0
