#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=finetune_output_%j.txt
#SBATCH --error=finetune_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1

bash run_hparam_optimization.sh --exp_name 'ShallowConvNet_BNCI2014001_hopt' \
                            --output_folder results/BNCI2014001/ShallowConvNet/hopt \
                            --data_folder data \
                            --cached_data_folder data \
                            --hparams hparams/MotorImagery/BNCI2014001/ShallowConvNet.yaml \
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
                            --project 'ShallowConvNet_BNCI2014001_hopt' \
                            --mode 'online'