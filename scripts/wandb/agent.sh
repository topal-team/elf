#!/bin/bash

if [ $# -eq 1 ] ; then
    c=$1
else
    c=5
fi

for i in $(seq 1 $c) ; do
    wandb agent --count=1 topal-inria/metis-estimate/sx2qwgf7 &
done

wait $(jobs -rp) && wandb sync --sync-all
