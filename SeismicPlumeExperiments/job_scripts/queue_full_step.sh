#!/bin/bash -e

if [ $# != 6 ]; then
    echo "usage: $0 params_file job_dir step_index num_trans_jobs num_obs_jobs run_dependency"
    echo "  the run_dependency can be 'nothing' or something like 'afterok:8744'"
    exit 1
fi
# For example.
params_file="run_test/params.toml"
job_dir="run_test"
step_index="0"
input_dep="nothing"
num_trans="4"
num_obs="8"

params_file="$1"
job_dir="$2"
step_index="$3"
num_trans="$4"
num_obs="$5"
input_dep="$6"

use_array="array"

if [ "$input_dep" = "nothing" ]; then
    input_dep=""
fi

if [[ `hostname` == "cruyff" ]]; then
    script_dir="./job_scripts/cruyff"
elif [[ `hostname` == *pace.gatech.edu ]]; then
    script_dir="./job_scripts/pace"
else
    echo "Error: unknown hostname `hostname`"
    exit 1
fi

# 0. Initialize samples.

# 1. Run transition function on every sample.
#   - Read inputs input from a file.
#   - Save output states to a file.
# 2. Run observation function on every sample.
#   - Read PDE states.
#   - Save RTMs to a file.
# 3. Update every sample based on a ground-truth observation.
#   - Read RTMs.
#   - Save states to a file.

if [ "$step_index" = "0" ]; then
    jid_final=`"$script_dir"/filter_initialize.sh "$params_file" "$job_dir" "$input_dep"`
    echo "$jid_final"
else
    jid_trans_final=`./job_scripts/queue_transition.sh "$params_file" "$job_dir" "$step_index" "$num_trans" "$input_dep" "$use_array"`

    input_dep="afterok:$jid_trans_final"
    jid_obs_final=`./job_scripts/queue_observe.sh "$params_file" "$job_dir" "$step_index" "$num_obs" "$input_dep" "$use_array"`

    # Launch one filter.
    input_dep="afterok:$jid_obs_final"
    jid_filter=`"$script_dir"/filter_assimilate.sh  "$params_file" "$job_dir" "$step_index" "$input_dep"`

    # Launch one filter.
    input_dep="afterok:$jid_filter"
    jid_process=`"$script_dir"/filter_process_assimilation.sh  "$params_file" "$job_dir" "$step_index" "$input_dep"`
    echo "$jid_process"
fi
