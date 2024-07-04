#!/bin/bash

if [ $# -eq 1 ] ; then
    c=$1
else
    c=5
fi

wandb agent --count=$c topal-inria/metis-estimate/whsk7pdy
