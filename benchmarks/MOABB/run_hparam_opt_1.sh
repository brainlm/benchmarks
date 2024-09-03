#!/bin/bash
#SBATCH --job-name=STGNN_Lee2019_MI
#SBATCH --output=STGNN_Lee2019_MI_output_%j.txt
#SBATCH --error=STGNN_Lee2019_MI_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1

# Mofidy this with the data change
output_folder="results/Lee2019_MI/STGNN/hopt"
final_yaml_file="$output_folder/best_hparams.yaml"
data_folder="data"
nsbj=54
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
                      --project 'STGNN_Lee2019_MI_eval' \
                      --mode 'online'

echo "The test performance with best hparams is available at  $output_folder/best"

