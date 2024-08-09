#!/bin/bash

# Define the array of numbers
numbers=(17 13 11 9 4)

set -e
# Iterate over the array
for k in "${numbers[@]}"; do
    # Execute the run_experiments.sh script with the specified parameters
    ./run_experiments.sh --hparams "hparams/MotorImagery/BNCI2014001/SpatialEEGNet.yaml" \
    --data_folder "~/mne_data" \
    --cached_data_folder "~/mne_data/pkl" \
    --output_folder "results/MotorImagery/BNCI2014001/SpatialEEGNet/eval/choose_k$k" \
    --nsbj 9 \
    --nsess 2 \
    --nruns 10 \
    --train_mode "leave-one-session-out" \
    --channel_drop_choose_k "$k" \
    --device "cuda"
done
