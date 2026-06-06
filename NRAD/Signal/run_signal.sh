#!/bin/bash
# Run Analyzer.C for selected signal modes, excluding some combinations
input_base="/home/aegis/ether/replicate_deepak"
# input_base="/home/aegis/Titan0/madgraph/MG5_aMC_v3_7_0/bin/ttbar_validation/Events/BSM_events"
output_base="/home/aegis/ether/Research_HEP/NRAD/Signal/replicate_atlas"
# Define parameter sets


tags=("1000_6" "1000_8" "2000_4" "2000_6" "3000_2" "3000_4")
versions=("I" "II" "III" "IV" "V" "VI" "VII" "VIII" "IX" "X" "XI")

for ver in "${versions[@]}"; do
  echo ">>> Processing version ${ver}"
  for tag in "${tags[@]}"; do
    IFS='_' read -r m r <<< "$tag"
    echo ">>> Processing m=${m}, r=${r}"
    input_file="${input_base}/${m}_${r}_events_mark${ver}.root"
    output_file="${output_base}/${m}_${r}_events_test${ver}.txt"
    root -l -b -q "Analyzer_BSM.C(\"${input_file}\",\"${output_file}\")"
  done
done

echo "All selected jobs finished."
