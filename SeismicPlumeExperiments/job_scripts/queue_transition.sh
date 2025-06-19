#!/bin/bash -e

if [ $# != 6 ]; then
    echo "usage: $0 params_file job_dir step_index num_jobs run_dependency use_array"
    echo "  the run_dependency can be 'nothing' or something like 'afterok:8744'"
    exit 1
fi

params_file="$1"
job_dir="$2"
step_index="$3"
num_jobs="$4"
input_dep="$5"
use_array="$6"

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

# Launch multiple helper transitions and one closer.
dep="afterany"
if [ "$num_jobs" -eq 0 ]; then
    dep="$input_dep"
elif [ "$use_array" = "array" ]; then
    jid=`"$script_dir"/filter_transition_array.sh  "$params_file" "$job_dir" "$step_index" helper "$num_jobs" "$input_dep"`
    dep="$dep:$jid"
elif [ "$use_array" = "no_array" ]; then
    for i in `seq $num_jobs`; do
        jid=`"$script_dir"/filter_transition.sh  "$params_file" "$job_dir" "$step_index" helper-$i-$num_jobs "$input_dep"`
        dep="$dep:$jid"
    done
else
    echo "Invalid use_array value: '$use_array'"
    exit 1
fi
jid_final=`"$script_dir"/filter_transition.sh  "$params_file" "$job_dir" "$step_index" closer "$dep"`
echo "$jid_final"
