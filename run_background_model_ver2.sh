#!/bin/bash
#SBATCH --job-name=mk_bkg_v2
#SBATCH --partition=tartarus
#SBATCH --output=background_model_ver2.out   # stdout (%j = job ID)
#SBATCH --error=background_model_ver2.err    # stderr
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# Move into project directory
cd /home/aegis/ether/Research_HEP

# Run background_model_ver2 on all variables (non-interactive)
python3 background_model_ver2.py -var all
