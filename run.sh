module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

python -m scripts.train_semseg --name segmentation_tiest_100 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --use_wandb True --num_epochs 100
