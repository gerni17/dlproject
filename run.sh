module load gcc/8.2.0
module load python_gpu
module load eth_proxy

python -m scripts.train_gogoll --name gogol_test --log_dir /cluster/scratch/gerni/logs --dataset_root ./data --use_wandb True