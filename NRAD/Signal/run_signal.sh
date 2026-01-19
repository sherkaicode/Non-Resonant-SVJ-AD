#!/bin/bash
# Run Analyzer.C for selected signal modes, excluding some combinations
input_base="/home/aegis/ether/replicate_deepak"
output_base="/home/aegis/ether/Research_HEP/NRAD/Signal/replicate_atlas"
# Define parameter sets


tags=("1000_6" "1000_8" "2000_4" "2000_6" "3000_2" "3000_4")

for tag in "${tags[@]}"; do
  IFS='_' read -r m r <<< "$tag"
  echo ">>> Processing m=${m}, r=${r}"
  input_file="${input_base}/${m}_${r}_events_markI.root"
  output_file="${output_base}/${m}_${r}_events_markI.txt"
  root -l -b -q "Analyzer.C(\"${input_file}\",\"${output_file}\")"
done

# Base directories
# input_base="/home/aegis/Titan0/pythia2/pythia8245/examples/Research/root_files_PU_lxplus"
# input_base="/home/aegis/ether/replicate_deepak"
# input_base="/home/aegis/ether/root_files_PU_lxplus"
# input_base="/home/aegis/ether/root_files_NPU_lxplus"
# input_base="/home/aegis/ether/root_files_100_NPU_lxplus"
# output_base="."
# output_base="/home/aegis/ether/Research_HEP/NRAD/Signal/lxplus_signal"
# output_base="/home/aegis/ether/Research_HEP/NRAD/Signal/replicate_deepak"

# Create output directory if it doesn't exist
# mkdir -p "$output_base"

# # Loop through combinations
# for m in "${masses[@]}"; do
#   for r in "${ratios[@]}"; do

#     # Skip unwanted combinations
#     # if { [ "$m" -eq 3000 ] && [ "$r" -eq 8 ]; }; then
#       #  { [ "$m" -eq 2000 ] && [ "$r" -eq 8 ]; } || \
#       #  { [ "$m" -eq 3000 ] && [ "$r" -eq 6 ]; }; then
#       # echo ">>> Skipping m=${m}, r=${r}"
#       # continue
#     # fi

#     # Build input/output paths
#     input_file="${input_base}/${m}_${r}_events.root"
#     output_file="${output_base}/${m}_${r}_events.txt"

#     echo ">>> Processing m=${m}, r=${r}"
#     root -l -b -q "Analyzer.C(\"${input_file}\",\"${output_file}\")"
#   done
# done

echo "All selected jobs finished."
