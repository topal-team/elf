#!/bin/bash

# Define arrays of hyperparameters
nodes=(2 4 8 16 32)
#dps=(1 2 4 8 16)
dps=(1 2)
archs=(GPT13B)
batch_sizes=(32 64 128)
sheds=('1f1b' 'afab' 'megatron')

SCRATCH="/net/home/project/tutorial/tutorial050/"
SBATCHSCRIPT_PATH=$SCRATCH/topal-internship/train_llama.sbatch
GROUPS_STORAGE="/net/storage/pr3/project/tutorial/elf/models"

generate_placement() {
  local pp=$1
  local sched=$2
  local placement=""
  # Generate basic sequence [0, 1, ..., pp-1]
  for i in $(seq 0 $((pp - 1))); do
    placement+="$i "
  done
  # If schedule is 'megatron', duplicate the sequence
  if [ "$sched" == "megatron" ]; then
    placement+="$placement"
  fi
  # Format the placement string as a list
  echo "[${placement% }]"
}

# Loop over each parameter combination
for arch in "${archs[@]}"; do
  for node in "${nodes[@]}"; do
    for dp in "${dps[@]}"; do
      if (( node % dp == 0 )); then
        pp=$((node / dp))
        # Skip if pp is less than 4
        if (( pp < 4 )); then
          continue
        fi
      for sched in "${sheds[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            placement=$(generate_placement "$pp" "$sched")
            if [ "$sched" == "megatron" ]; then
                sched="1f1b"
            fi
            savedir="${GROUPS_STORAGE}/elf/models/$"
            # Create a single argument string
            PYARGS="--model gpt --phase train --save_dir $savedir --gpt.arch $arch --pipeline.dp $dp --pipeline.pp $pp --pipeline.pipeline_sched $sched --pipeline.pp_placement $placement --train.batch_size $bs"
            echo $PYARGS
            # Submit the job with sbatch, passing the argument string as one variable
            sbatch --nodes $node --gres=gpu:4 --export=PYARGS="${PYARGS}" ${SBATCHSCRIPT_PATH}
        done
      done
      fi
    done
  done
done