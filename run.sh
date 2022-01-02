module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

<<<<<<< HEAD
# python -m euler_run_baselines --name baselines_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False
# python -m euler_run_cross_val --name baselines_gogoll_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False
# python -m euler_run_gam --name gam_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_gogoll 900
python -m euler_run_cycle_gan --name cycle_gan --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_gogoll 50
=======
bash download_euler /cluster/scratch/$USER
>>>>>>> 97dc1f8aa566032cbcf08118df22ec33000d8fde


# run semseg
python -m scripts.run_semseg --name semseg

# run baselines
python -m scripts.run_baselines --name baselines 

# run cycle gan
python -m scripts.run_cyclegan --name cycle 

# run gogoll 
python -m scripts.run_gogoll --name gogoll

# run gogoll cross validation
python -m scripts.run_gogoll_crossval --name gogoll

# run gam 
python -m scripts.run_gam --name gam

# run embedding
python -m scripts.run_gam --name embedding

<<<<<<< HEAD
# python -m euler_run_cross_val --name entire_pipe_segw_5 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --segmentation_weight 0.5 --shared True
# python -m euler_run_cross_val --name entire_pipe_segw_6 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --segmentation_weight 0.6 --shared True
# python -m euler_run_cross_val --name entire_pipe_segw_8 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --segmentation_weight 0.8 --shared True
=======
>>>>>>> 97dc1f8aa566032cbcf08118df22ec33000d8fde

