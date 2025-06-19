#!/bin/bash

if [ $# -lt 5 ] || [ $# -gt 6 ]; then
    echo "usage: $0 params_file job_dir step_index worker_type num_jobs [dependency_string]"
    echo
    echo "The dependency string should look something like ""afterok:7475"", but you could probably do"
    echo " command injection with it."
    exit 1
fi

params_file="$1"
job_dir="$2"
step_index="$3"
worker_type="$4"
num_jobs="$5"
dependency_string="$6"

if [[ "$worker_type" != "helper" ]] && [[ "$worker_type" != 'closer' ]]; then
    echo "worker type has to be 'helper' or 'closer': you put '$worker_type'"
    exit 1
fi

if [[ "$worker_type" = "closer" ]] && [[ "$num_jobs" != 1 ]]; then
    echo "closer must have exactly 1 job: you put '$num_jobs'"
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
max_running=2

$_command << EOT
#!/bin/bash

#SBATCH --job-name=obs"$step_index"
#SBATCH -o "$job_dir"/slurm/slurm-out-obs"$step_index"-%A_%a.txt
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:"$ngpus"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task="$num_cpus"
#SBATCH --mem-per-cpu=8G
#SBATCH --array 1-"$num_jobs"%"$max_running"

module load Julia/1.8/5 cudnn nvhpc

export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

export OMP_NUM_TREADS=1
export JULIA_NUM_THREADS=1

worker_type_fancy="$worker_type"
if [[ "\$worker_type_fancy" = "helper" ]]; then
    worker_type_fancy="helper-\$SLURM_ARRAY_TASK_ID-\$SLURM_ARRAY_TASK_COUNT"
fi
echo "\$worker_type_fancy"

/usr/bin/time --verbose srun julia --project=. -p "$num_processes" job_scripts/parallel_filter_observe.jl "$params_file" "$job_dir" "$step_index" "\$worker_type_fancy"
EOT
