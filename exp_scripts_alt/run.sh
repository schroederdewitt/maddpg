#!/bin/bash

target=$1
cmd_line=$2
tag=$3
reps=$4

filename=$(basename -- "$0")
extension="${filename##*.}"
filename="${filename%.*}"
timestamp=`date "+%d_%m_%y-%H_%M_%S"`
name=${tag}__${filename}__${timestamp}

# set up experiment summary file
spath="${MADDPG_PATH}/exp_summaries"
mkdir -p $spath
sfilepath="$spath/${tag}.summary"
if [ -f $sfilepath ]; then
    echo "FATAL: experiment summary already exists. Overwrite?"
    read -p "Are you sure? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "Aborting run."
        exit
    else
        echo "Deleting old experiment summary..."
        rm -rf $sfilepath
    fi
fi

# set pymarl results path
if [ -z "$MADDPG_RESULTS_PATH" ]; then
    RESULTS_PATH="${MADDPG_PATH}/results"
    mkdir -p $RESULTS_PATH
else
    RESULTS_PATH=$MADDPG_RESULTS_PATH
fi

if [ $target == "local" ] ; then

    echo "launching locally on "`hostname`"..."
    export PYTHONPATH=$PYTHONPATH:/maddpg

    # enter general run information into summary file
    echo "hostname: "`hostname`" "
    echo "pymarl_path: ${MADDPG_PATH}" >> $sfilepath
    echo "python_path: ${PYTHONPATH}" >> $sfilepath
    echo "results_path: ${RESULTS_PATH}" >> $sfilepath

    #if [ `hostname` == "octavia" | `hostname` ==  ]

    n_gpus=`nvidia-smi -L | wc -l`
    n_upper=`expr $n_gpus - 1`

    for i in $(seq 1 $reps); do
        gpu_id=`shuf -i0-${n_upper} -n1`
        echo "Starting repeat number $i on GPU $gpu_id"
        HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
        echo "NV_GPU=${gpu_id} ${MADDPG_PATH}/docker.sh ${HASH} python3 train.py ${cmd_line} --exp-name ${name}__repeat${i} &"
        NV_GPU=${gpu_id} ${MADDPG_PATH}/docker.sh ${HASH} python3 train.py ${cmd_line} --exp-name ${name}__repeat${i} &
        echo "repeat: ${i}"
        echo "    name: ${name}__repeat${i}" >> $sfilepath
        echo "    gpu: ${gpu_id}" >> $sfilepath
        echo "    docker_hash: ${HASH}" >> $sfilepath
        sleep 5s
    done

else

    echo "Target ${target} not supported!"
    exit

fi

echo "Finished experiment launch on "`hostname`"."
