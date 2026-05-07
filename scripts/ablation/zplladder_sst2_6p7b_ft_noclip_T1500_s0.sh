#!/bin/bash
# PAC-ZPL canonical 6.7B FT T=1500, seed=0
# Paper reference: appendix.tex Tables zpl-67b-ft-ladder + zpl-67b-ft-multiseed
# Recipe: ft no-clip lr=1e-4 c=1e9 T=1500 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl zplladder_sst2_6p7b_ft_noclip_T1500_s0 facebook/opt-6.7b SST2 ft 1e-4 1e9 1500 128 0
