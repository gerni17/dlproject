module load gcc/8.2.0
module load python_gpu
module load eth_proxy

python -m euler_run_cross_val --name whole_pipeline_test --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dat/data --use_wandb True  
# python -m scripts.train_cross_val --name whole_pipeline_test --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dat/data --use_wandb True  --seg_checkpoint_path /cluster/scratch/gerni/logs/whole_pipeline_test_1201-1709_34/segmentation/'epoch=15-step=847.ckpt' --gogoll_checkpoint_path /cluster/scratch/gerni/logs/whole_pipeline_test_1201-1709_34/gogoll/'epoch=73-step=3921.ckpt'
