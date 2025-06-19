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

ngpus=0
num_processes=1
num_cpus=2

src_file="job_scripts/parallel_filter_observe.jl"

$_command << EOT
#!/bin/bash

#SBATCH --job-name=obs"$step_index"
#SBATCH -o "$job_dir"/slurm/slurm-out-obs"$step_index"-%j.txt
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:"$ngpus"
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task="$num_cpus"
#SBATCH --mem-per-cpu=8G
#SBATCH -q inferno
#SBATCH -A gts-echow7

module purge
module load gcc julia/1.8.0

# export DEVITO_LANGUAGE=openmp
# export DEVITO_ARCH=gcc

export OMP_NUM_TREADS=1
export JULIA_NUM_THREADS=1

/usr/bin/time --verbose srun julia -p 1 "$src_file" "$params_file" "$job_dir" "$step_index" "$worker_type"
EOT

sleep 1
