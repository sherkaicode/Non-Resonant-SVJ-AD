#!/bin/bash
# Run Analyzer.C for selected signal modes, excluding some combinations
input_base="/home/aegis/ether/replicate_deepak"
output_base="/home/aegis/ether/Research_HEP/NRAD/Signal/replicate_atlas"
# Define parameter sets


tags=("1000_6" "1000_8" "2000_4" "2000_6" "3000_2" "3000_4")

for tag in "${tags[@]}"; do
  IFS='_' read -r m r <<< "$tag"
  echo ">>> Processing m=${m}, r=${r}"
  input_file="${input_base}/${m}_${r}_events_markXI.root"
  output_file="${output_base}/${m}_${r}_events_markXI.txt"
  root -l -b -q "Analyzer_BSM.C(\"${input_file}\",\"${output_file}\")"
done

echo "All selected jobs finished."
