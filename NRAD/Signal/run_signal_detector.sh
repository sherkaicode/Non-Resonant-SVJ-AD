#!/bin/bash
# Run Analyzer.C for all detector-level ROOT files in the folder

# Parameter sets
masses=(1000 2000 3000)
ratios=(2 4 6 8)

# Base directories
input_base="/home/aegis/ether/root_files_final_no_muons"
output_base="final_signals_no_muons"

# Create output directory if it doesn't exist
mkdir -p "$output_base"

# Loop through combinations
for m in "${masses[@]}"; do
  for r in "${ratios[@]}"; do

    echo ">>> Searching for files: ${m}_${r}__*.root"

    # Loop through all matching ROOT files
    for input_file in "${input_base}/${m}_${r}_"*.root; do

      # Skip if no file was found
      if [[ ! -e "$input_file" ]]; then
        echo "No files found for m=$m, r=$r"
        continue
      fi

      # Extract filename without path
      filename=$(basename "$input_file")

      # Create matching output filename (.txt)
      output_file="${output_base}/${filename%.root}.txt"

      echo ">>> Processing: $filename"
      root -l -b -q "Analyzer.C(\"${input_file}\",\"${output_file}\")"

    done

  done
done

echo "All jobs finished."
