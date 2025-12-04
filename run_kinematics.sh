#!/bin/bash
#SBATCH --job-name=mk_kinematics
#SBATCH --partition=tartarus
#SBATCH --output=kinematics.out   # stdout (%j = job ID)
#SBATCH --error=kinematics.err    # stderr
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

source /home/aegis/NRAD_env/bin/activate
# Move into project directory
cd /home/aegis/ether/Research_HEP

# Run the plotting script (non-interactive, saves figures to plots/Data/kinematics)
python3 run_kinematics.py
