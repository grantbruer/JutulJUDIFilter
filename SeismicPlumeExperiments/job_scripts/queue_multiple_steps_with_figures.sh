#!/bin/bash -e

if [ $# != 7 ]; then
    echo "usage: $0 params_file job_dir step_index_start step_index_end num_trans_jobs num_obs_jobs run_dependency"
    echo "  the run_dependency can be 'nothing' or something like 'afterok:8744'"
    exit 1
fi
# For example.
params_file="run_more/params.toml"
job_dir="run_more"
step_index_start="14"
step_index_end="15"
input_dep="afterok:8696"
num_trans="3"
num_obs="6"

params_file="$1"
job_dir="$2"
step_index_start="$3"
step_index_end="$4"
num_trans="$5"
num_obs="$6"
input_dep="$7"

if [[ `hostname` == "cruyff" ]]; then
    script_dir="./job_scripts/cruyff"
elif [[ `hostname` == *pace.gatech.edu ]]; then
    script_dir="./job_scripts/pace"
else
    echo "Error: unknown hostname `hostname`"
    exit 1
fi

for step_index in `seq "$step_index_start" "$step_index_end"`; do
    jid_final=`./job_scripts/queue_full_step.sh "$params_file" "$job_dir" "$step_index" "$num_trans" "$num_obs" "$input_dep"`
    input_dep="afterok:$jid_final"
done

echo "$jid_final"

"$script_dir"/filter_figures_plumes.sh "$params_file" "$job_dir" "$input_dep"
"$script_dir"/filter_figures_errors.sh "$params_file" "$job_dir" "$input_dep"
