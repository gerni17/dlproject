
module load gcc/8.2.0
module load python_gpu
module load eth_proxy

# python -m scripts.train_semseg --name semseg --log_dir /cluster/scratch/gerni/logs --dataset_root ./data --num_epochs 100 --use_wandb True
python -m scripts.train_gogolgan --name gogol_test --log_dir /cluster/scratch/gerni/logs --dataset_root ./data --num_epochs 170 --use_wandb True --resume /cluster/scratch/gerni/logs/semseg_1120-0107_79/epoch\=96-step\=19399.ckpt\
