module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

python -m euler_run_cross_val --name embedding_L2_0_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type L2 --w_embed 0.3
python -m euler_run_cross_val --name embedding_L2_1 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type L2 --w_embed 1
python -m euler_run_cross_val --name embedding_L2_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type L2 --w_embed 3


python -m euler_run_cross_val --name embedding_L1_0_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type L1 --w_embed 0.3
python -m euler_run_cross_val --name embedding_L1_1 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type L1 --w_embed 1
python -m euler_run_cross_val --name embedding_L1_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type L1 --w_embed 3


python -m euler_run_cross_val --name embedding_cosine_0_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type cosine --w_embed 0.3
python -m euler_run_cross_val --name embedding_cosine_1 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type cosine --w_embed 1
python -m euler_run_cross_val --name embedding_cosine_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True --loss_type cosine --w_embed 3



