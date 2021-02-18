#!/bin/bash

if [ -z "$3" ]; then
    echo "target 'local' selected automatically."
    target="local"
    tag=$1
    reps=$2
else
   target=$1
   tag=$2
   reps=$3
fi 

cmd_line=" --scenario simple_spread --plots-dir /maddpg/results/plots --benchmark-dir /maddpg/results/benchmark"

${MADDPG_PATH}/exp_scripts_alt/run.sh "${target}" "${cmd_line}" "${tag}" "${reps}"
