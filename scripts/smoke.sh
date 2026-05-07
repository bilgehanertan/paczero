#!/bin/bash
# 1-2 minute smoke check: PAC-MI on SST-2 1.3B LoRA at SMOKE_STEPS=5.
SMOKE=1 SMOKE_STEPS=5 SMOKE_EVAL=5 \
  bash "$(dirname "${BASH_SOURCE[0]}")/headline/sst2_1p3b_lora_pacmi_mi033_s0.sh"
