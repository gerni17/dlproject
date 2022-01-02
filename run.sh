module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy

bash download_euler /cluster/scratch/$USER


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


