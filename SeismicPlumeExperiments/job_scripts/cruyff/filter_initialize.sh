#!/bin/bash

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "usage: $0 params_file job_dir [dependency_string]"
    echo
    echo "The dependency string should look something like ""afterok:7475"", but you could probably do"
    echo " command injection with it."
    exit 1
fi

params_file="$1"
job_dir="$2"
dependency_string="$3"

_command="sbatch --parsable"
if [ -n "$dependency_string" ]; then
    _command="$_command --dependency=$dependency_string"
fi

mkdir -p "$job_dir"/slurm

$_command << EOT
#!/bin/bash

#SBATCH --job-name=initialize"$step_index"
#SBATCH -o "$job_dir"/slurm/slurm-out-init-%j.txt
#SBATCH --time=8:00:00
#SBATCH -p cpu
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

module load Julia/1.8/5

export OMP_NUM_TREADS=4
export JULIA_NUM_THREADS=4

/usr/bin/time --verbose srun julia  scripts/filter_initialize.jl "$params_file" "$job_dir"
EOT
