module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

bash download_euler.sh /cluster/scratch/$USER


# run semseg
python -m scripts.run_semseg --name semseg

# run baselines
python -m scripts.run_baselines --name baselines 

# run cycle gan
python -m scripts.run_cycle_gan --name cycle 

# run gogoll gan
python -m scripts.run_gogoll_gan --name gogoll

# run labeltotarget
python -m scripts.run_labeltotarget --name labeltotarget

# run attention gan
python -m scripts.run_attention_gan --name attention 

# run embedding
python -m scripts.run_embedding --name embedding

# run stacking
python -m scripts.run_stacking --name stacking
