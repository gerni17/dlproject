module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

if [ ! -d /cluster/scratch/$USER/dl_data ]; then
  /bin/bash download_euler.sh /cluster/scratch/$USER/dl_data
fi

# run semseg
python -m scripts.run_semseg --name semseg --num_epochs_seg 2

# run baselines
python -m scripts.run_baselines --name baselines --num_epochs_final 2 

# run cycle gan
python -m scripts.run_cycle_gan --name cycle --num_epochs_seg 1 --num_epochs_final 2 --num_epochs_cyclegan 2

# run gogoll gan
python -m scripts.run_gogoll_gan --name gogoll --num_epochs_seg 1 --num_epochs_final 2 --num_epochs_gogoll 2

# run labeltotarget
python -m scripts.run_labeltotarget --name labeltotarget --num_epochs_final 2 --num_epochs_labeltotarget 2

# run attention gan
python -m scripts.run_attention_gan --name attention --num_epochs_seg 1 --num_epochs_final 2 --num_epochs_gogoll 2

# run embedding
python -m scripts.run_embedding --name embedding --num_epochs_seg 1 --num_epochs_final 2 --num_epochs_gogoll 2

# run stacking
python -m scripts.run_stacking --name stacking --num_epochs_seg 1 --num_epochs_final 2 --num_epochs_gogoll 2
