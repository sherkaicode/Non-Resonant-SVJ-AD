#!/bin/bash
# Run Analyzer.C for selected signal modes, excluding some combinations

# Define parameter sets
masses=(1000 2000 3000)
ratios=(2 4 6 8)

# Base directories
input_base="/Titan0/aegis/pythia2/pythia8245/examples/Research/root_files_edit"
output_base="processed_signal"

# Create output directory if it doesn't exist
mkdir -p "$output_base"

# Loop through combinations
for m in "${masses[@]}"; do
  for r in "${ratios[@]}"; do

    # Skip unwanted combinations
    if { [ "$m" -eq 3000 ] && [ "$r" -eq 8 ]; }; then
      #  { [ "$m" -eq 2000 ] && [ "$r" -eq 8 ]; } || \
      #  { [ "$m" -eq 3000 ] && [ "$r" -eq 6 ]; }; then
      echo ">>> Skipping m=${m}, r=${r}"
      continue
    fi

    # Build input/output paths
    input_file="${input_base}/${m}_${r}_events.root"
    output_file="${output_base}/${m}_${r}_events.txt"

    echo ">>> Processing m=${m}, r=${r}"
    root -l -b -q "Analyzer.C(\"${input_file}\",\"${output_file}\")"
  done
done

echo "All selected jobs finished."
