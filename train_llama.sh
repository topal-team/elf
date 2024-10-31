#!/bin/bash -l

SBATCHSCRIPT_PATH="$HOME/topal-internship/train_llama.sbatch"
PYSCRIPT_PATH="$HOME/topal-internship/train_llama.py"
GROUPS_STORAGE='/net/storage/pr3/project/tutorial'

# Define arrays of hyperparameters
nodes=(4)
dps=(4)
archs=(GPTSmall)
batch_sizes=(32)
sheds=('afab')

generate_placement() {
  local pp=$1         # Number of pipeline parallel stages
  local sched=$2      # Schedule type, e.g., "megatron"
  local placement=""
  
  # Generate basic sequence [0, 1, ..., pp-1]
  for i in $(seq 0 $((pp - 1))); do
    placement+="$i,"
  done
  
  # If schedule is 'megatron', duplicate the sequence
  if [ "$sched" == "megatron" ]; then
    placement+="$placement"
  fi
  echo "${placement%,}"
}


# Loop over each parameter combination
for arch in "${archs[@]}"; do
  savedir=${GROUPS_STORAGE}/elf/models/$arch/
  for node in "${nodes[@]}"; do
    for dp in "${dps[@]}"; do
      if (( node % dp == 0 )); then
        pp=$((node * 4 / dp))
        # Skip if pp is less than 4
        if (( pp < 4 )); then
          continue
        fi
      for sched in "${sheds[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            placement=$(generate_placement "$pp" "$sched")
            if [ "$sched" == "megatron" ]; then
                sched='1f1b'
            fi
            PYARGS="${PYSCRIPT_PATH} --model gpt --phase train --save_dir ${savedir} --gpt.arch $arch --pipeline.dp $dp --pipeline.pp $pp --pipeline.schedule_type $sched --pipeline.pp_placement $placement --train.batch_size $bs"
            echo $PYARGS
            # Submit the job with sbatch, passing the argument string as one variable
            # sbatch --nodes $node --ntasks-per-node 1 --gpus-per-task 1 
            sbatch --nodes $node --gres=gpu:4 ${SBATCHSCRIPT_PATH} $PYARGS
        done
      done
      fi
    done
  done
done