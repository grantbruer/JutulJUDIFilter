#!/bin/bash

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "usage: $0 params_file job_dir step_index worker_type [dependency_string]"
    echo
    echo "The dependency string should look something like ""afterok:7475"", but you could probably do"
    echo " command injection with it."
    exit 1
fi

params_file="$1"
job_dir="$2"
step_index="$3"
worker_type="$4"
dependency_string="$5"

if [[ ! "$worker_type" =~ ^helper-[0-9]+-[0-9]+$ ]] && [[ "$worker_type" != 'closer' ]]; then
    echo "worker type has to be 'helper-X-Y' or 'closer': you put '$worker_type'"
    exit 1
fi

_command="sbatch --parsable"
if [ -n "$dependency_string" ]; then
    _command="$_command --dependency=$dependency_string"
fi

mkdir -p "$job_dir"/slurm

ngpus=1
num_processes=1
num_cpus=2

if [ "$ngpus" -eq 0 ]; then
    partition="cpu"
else
    partition="gpu"
fi
src_file="job_scripts/parallel_filter_observe.jl"

$_command << EOT
#!/bin/bash

#SBATCH --job-name=obs"$step_index"
#SBATCH -o "$job_dir"/slurm/slurm-out-obs"$step_index"-%j.txt
#SBATCH --time=48:00:00
#SBATCH -p "$partition"
#SBATCH --gres=gpu:"$ngpus"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task="$num_cpus"
#SBATCH --mem-per-cpu=8G

module load Julia/1.8/5 cudnn nvhpc

if [ "$ngpus" -gt 0 ]; then
    export DEVITO_LANGUAGE=openacc
    export DEVITO_ARCH=nvc
    export DEVITO_PLATFORM=nvidiaX
fi

export OMP_NUM_TREADS=1
export JULIA_NUM_THREADS=1

/usr/bin/time --verbose srun julia -p "$num_processes" "$src_file" "$params_file" "$job_dir" "$step_index" "$worker_type"
EOT
