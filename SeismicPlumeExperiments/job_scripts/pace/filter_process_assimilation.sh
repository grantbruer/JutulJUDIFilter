#!/bin/bash

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: $0 params_file job_dir step_index [dependency_string]"
    echo
    echo "The dependency string should look something like ""afterok:7475"", but you could probably do"
    echo " command injection with it."
    exit 1
fi

params_file="$1"
job_dir="$2"
step_index="$3"
dependency_string="$4"

_command="sbatch --parsable"
if [ -n "$dependency_string" ]; then
    _command="$_command --dependency=$dependency_string"
fi

mkdir -p "$job_dir"/slurm

$_command << EOT
#!/bin/bash

#SBATCH --job-name=post"$step_index"
#SBATCH -o "$job_dir"/slurm/slurm-out-post"$step_index"-%j.txt
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G
#SBATCH -q inferno
#SBATCH -A gts-echow7

module purge
module load gcc julia/1.8.0

export OMP_NUM_TREADS=1
export JULIA_NUM_THREADS=1

/usr/bin/time --verbose srun julia scripts/filter_process_assimilation.jl "$params_file" "$job_dir" "$step_index"
EOT

sleep 1
