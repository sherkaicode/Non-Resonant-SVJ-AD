#!/bin/bash
#SBATCH --job-name=mk_data_I
#SBATCH --partition=tartarus
#SBATCH --output=data_I.out   # stdout (%j = job ID)
#SBATCH --error=data_I.err    # stderr
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# Move into project directory
cd /home/aegis/ether/Research_HEP

# Run Python script
python3 make_data_script.py -period I
