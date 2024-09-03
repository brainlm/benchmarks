#!/bin/bash
#SBATCH --job-name=STGNN_BNCI2014001
#SBATCH --output=STGNN_BNCI2014001_output_%j.txt
#SBATCH --error=STGNN_BNCI2014001_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=35:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1

bash run_hparam_optimization.sh --exp_name 'BNCI2014001_hopt_2' \
                            --output_folder results/BNCI2014001/STGNN/hopt_2 \
                            --data_folder data \
                            --hparams  hparams/MotorImagery/BNCI2014001/STGNN.yaml \
                            --nsbj_hpsearch 9 --nsess_hpsearch 2 \
                            --nsbj 9 --nsess 2 \
                            --nruns 1 --nruns_eval 0 \
                            --eval_metric acc \
                            --train_mode leave-one-subject-out \
                            --exp_max_trials 50 \
                            --store_all True \
                            --device 'cuda' \
                            --project 'BNCI2014001_hopt_2' \
                            --mode 'online'