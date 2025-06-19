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

ngpus=1

$_command << EOT
#!/bin/bash

#SBATCH --job-name=assimilate"$step_index"
#SBATCH -o "$job_dir"/slurm/slurm-out-assimilate"$step_index"-%j.txt
#SBATCH --time=8:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:"$ngpus"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G

module load Julia/1.8/5 cudnn nvhpc

export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

export OMP_NUM_TREADS=2
export JULIA_NUM_THREADS=2

/usr/bin/time --verbose srun julia scripts/filter_assimilate.jl "$params_file" "$job_dir" "$step_index"
EOT
