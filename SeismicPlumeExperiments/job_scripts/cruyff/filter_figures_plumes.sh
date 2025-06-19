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

#SBATCH --job-name=fig-plumes
#SBATCH -o "$job_dir"/slurm/slurm-out-fig-plumes-%j.txt
#SBATCH --time=4:00:00
#SBATCH -p cpu
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

module load Julia/1.8/5 cudnn nvhpc

export OMP_NUM_TREADS=1
export JULIA_NUM_THREADS=1

/usr/bin/time --verbose srun julia scripts/filter_figures_plume.jl "$params_file" "$job_dir"
EOT
