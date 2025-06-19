#!/bin/bash
if [ $# != 1 ]; then
    echo "usage: $0 job_dir"
    exit 1
fi
date
echo
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
squeue -u `whoami` | grep " R"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo
for f in "$1"/*/intermediate*/; do
    echo -n "$f":"   "
    find "$f" -type f | wc -l
done
