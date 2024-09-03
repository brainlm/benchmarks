#!/bin/bash
#SBATCH --job-name=STGNN_BNCI2014001
#SBATCH --output=STGNN_BNCI2014001_output_%j.txt
#SBATCH --error=STGNN_BNCI2014001_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1

# Mofidy this with the data change
output_folder="results/BNCI2014001/STGNN/hopt_2"
final_yaml_file="$output_folder/best_hparams.yaml"
data_folder="data"
nsbj=9
nsess=2
nruns_eval=10
eval_metric="acc"
train_mode="leave-one-subject-out"
store_all=True

# Running evaluation on the test set for the best models
bash run_experiments.sh --hparams $final_yaml_file \
                      --data_folder $data_folder \
                      --output_folder $output_folder/best \
                      --nsbj $nsbj --nsess $nsess \
                      --nruns $nruns_eval \
                      --eval_metric $eval_metric \
                      --eval_set test \
                      --train_mode $train_mode \
                      --rnd_dir $store_all \
                      --device 'cuda' \
                      --number_of_epochs 100 \
                      --project 'STGNN_BNCI2014001_eval' \
                      --mode 'online'

echo "The test performance with best hparams is available at  $output_folder/best"
