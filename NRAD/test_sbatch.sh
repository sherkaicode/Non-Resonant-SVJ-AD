#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --partition=tartarus
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

echo "NODE: $(hostname)"

python3 - <<EOF
import numpy as np
import sys
print("NumPy version:", np.__version__)
print("NumPy file:", np.__file__)
print("sys.path:", sys.path)
EOF