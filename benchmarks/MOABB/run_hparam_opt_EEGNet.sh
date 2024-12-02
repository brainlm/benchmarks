#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=finetune_output_%j.txt
#SBATCH --error=finetune_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1

bash run_hparam_optimization.sh --exp_name 'EEGNet_BNCI2014001_hopt' \
                            --output_folder results/BNCI2014001/EEGNet/hopt \
                            --data_folder data \
                            --cached_data_folder data \
                            --hparams hparams/MotorImagery/BNCI2014001/EEGNet.yaml \
                            --nsbj_hpsearch 9 \
                            --nsess_hpsearch 2 \
                            --nsbj 9 \
                            --nsess 2 \
                            --nruns 1 \
                            --nruns_eval 10 \
                            --eval_metric acc \
                            --train_mode leave-one-subject-out \
                            --exp_max_trials 50 \
                            --store_all True \
                            --device 'cuda' \
                            --project 'EEGNet_BNCI2014001_hopt' \
                            --mode 'online'