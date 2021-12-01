module load gcc/8.2.0
module load python_gpu
module load eth_proxy

python -m scripts.train_gogoll --name whole_pipeline_test --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dat/data --use_wandb True