module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

# python -m euler_run_baselines --name baselines_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_seg 55 --num_epochs_gogoll 100 --num_epochs_final 16
# python -m euler_run_cross_val --name baselines_gogoll_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_seg 55 --num_epochs_gogoll 100 --num_epochs_final 16
# python -m euler_run_gam --name gam_withsched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared False --num_epochs_gogoll 900 --num_epochs_final 16

# python -m euler_run_cross_val --name gogol_epoch_100 --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data --use_wandb True --shared True \
# --num_epochs_gogoll 100 --num_epochs_seg 55
# python -m scripts.train_semseg --name trash --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched True --num_epochs_seg 1 --batch_size 8
# python -m scripts.train_semseg --name trash --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched False --num_epochs_seg 1 --batch_size 8

python -m scripts.train_semseg --name segmentation_test_16_sched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched True --num_epochs_seg 16 --batch_size 8
python -m scripts.train_semseg --name segmentation_test_55_sched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched True --num_epochs_seg 55 --batch_size 8
python -m scripts.train_semseg --name segmentation_test_16_no_sched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched False --num_epochs_seg 16 --batch_size 8
python -m scripts.train_semseg --name segmentation_test_55_no_sched --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dl_data/data/source --sched False --num_epochs_seg 55 --batch_size 8
