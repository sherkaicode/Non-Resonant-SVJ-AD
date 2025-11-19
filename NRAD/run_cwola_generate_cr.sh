#!/bin/bash
#SBATCH --job-name=generate_eval
#SBATCH --partition=tartarus
#SBATCH --output=generate.out
#SBATCH --error=generate.err
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

echo "Running on node:"
hostname

# ----------------------------------------------------
# 1. ACTIVATE CLEAN PYTHON ENVIRONMENT
# ----------------------------------------------------
echo "Activating NRAD_env..."
source $HOME/NRAD_env/bin/activate

# ----------------------------------------------------
# 2. SET YOUR CUSTOM HEP LIBRARIES
# ----------------------------------------------------
export LD_LIBRARY_PATH=$HOME/Titan0/programs/usr/lib:$LD_LIBRARY_PATH
export PATH=$HOME/Titan0/programs/usr/bin:$PATH
export X11_X11_INCLUDE_PATH=$HOME/Titan0/programs/usr/include:$X11_X11_INCLUDE_PATH
export X11_X11_LIB=$HOME/Titan0/programs/usr/lib:$X11_X11_LIB

export LD_LIBRARY_PATH=$HOME/Titan0/madgraph/MG5_aMC_v3_5_6/HEPTools/lhapdf6_py3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/Titan0/madgraph/MG5_aMC_v3_5_6/HEPTools/lhapdf6_py3/local/lib/python3.10/dist-packages:$PYTHONPATH

export MPLBACKEND=Agg

# ----------------------------------------------------
# 3. Debug: show Python + NumPy version inside SLURM
# ----------------------------------------------------
$HOME/NRAD_env/bin/python3 - <<EOF
import numpy as np, sys
print("Using Python:", sys.executable)
print("NumPy version:", np.__version__)
print("NumPy path:", np.__file__)
EOF

# ----------------------------------------------------
# 4. Run your script
# ----------------------------------------------------
echo "Starting Generate Evaluation..."
$HOME/NRAD_env/bin/python3 script/cwola_generate_cr.py
echo "Done."
