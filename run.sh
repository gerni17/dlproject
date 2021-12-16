module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

python -m euler_run_cross_val --name focal_loss_test --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False

#python -m euler_run_baselines --name baselines --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True
# python -m euler_run_cross_val --name baselines_ours --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dat/data --use_wandb True --shared False
