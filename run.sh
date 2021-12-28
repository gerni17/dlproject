module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

python -m euler_run_cross_val --name embedding_L2_0.3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type L2 --w_embed 0.3
python -m euler_run_cross_val --name embedding_L2_1 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type L2 --w_embed 1
python -m euler_run_cross_val --name embedding_L2_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type L2 --w_embed 3

python -m euler_run_cross_val --name embedding_L1_0.3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type L1 --w_embed 0.3
python -m euler_run_cross_val --name embedding_L1_1 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type L1 --w_embed 1
python -m euler_run_cross_val --name embedding_L1_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type L1 --w_embed 3

python -m euler_run_cross_val --name embedding_cos_0.3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type cosine --w_embed 0.3
python -m euler_run_cross_val --name embedding_cos_1 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type cosine --w_embed 1
python -m euler_run_cross_val --name embedding_cos_3 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --loss_type cosine --w_embed 3


# python -m euler_run_baselines --name baselines --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True
# python -m euler_run_cross_val --name baselines_ours --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dat/data --use_wandb True --shared False
# python -m euler_run_baselines --name baselines_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_seg 55 --num_epochs_gogoll 100 --num_epochs_final 16
# python -m euler_run_cross_val --name baselines_gogoll_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_seg 55 --num_epochs_gogoll 100 --num_epochs_final 16
# python -m euler_run_gam --name gam_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_gogoll 900 --num_epochs_final 16

# python -m euler_run_cross_val --name gogol_epoch_100 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True \
# --num_epochs_gogoll 100 --num_epochs_seg 55
# python -m scripts.train_semseg --name trash --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched True --num_epochs_seg 1 --batch_size 8
# python -m scripts.train_semseg --name trash --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched False --num_epochs_seg 1 --batch_size 8

# python -m scripts.train_semseg --name segmentation_test_16_sched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched True --num_epochs_seg 16 --batch_size 8

# python -m euler_run_cross_val --name entire_pipe_segw_5 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --segmentation_weight 0.5 --shared True
# python -m euler_run_cross_val --name entire_pipe_segw_6 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --segmentation_weight 0.6 --shared True
# python -m euler_run_cross_val --name entire_pipe_segw_8 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --segmentation_weight 0.8 --shared True

