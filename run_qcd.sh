#!/bin/bash
#SBATCH --job-name=mk_mc_QCD
#SBATCH --partition=tartarus
#SBATCH --output=mc_QCD.out   # stdout (%j = job ID)
#SBATCH --error=mc_QCD.err    # stderr
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# Move into project directory
cd /home/aegis/ether/Research_HEP

# Run Python script
python3 make_MC_script.py -process Multijet