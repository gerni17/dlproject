# Deep Learning Project

## Getting started

Make sure you install all required packages first (e.g wandb, pytorch, pytorch lightning etc)

To run the pipeline on euler run the following command in the dlproject directory:

```sh
bsub -n 2 -R "rusage[scratch=1000,mem=15000,ngpus_excl_p=1]" -oo /cluster/scratch/gerni/log < run.sh
```

To add commandline arguiments you can add them in the bash file run.sh e.g.:

```sh
python -m scripts.train_cross_val --name whole_pipeline_test --log_dir /cluster/scratch/$USER/logs --dataset_root /cluster/scratch/$USER/dat/data --use_wandb True  --seg_checkpoint_path /cluster/scratch/$USER/logs/whole_pipeline_test_1201-1709_34/segmentation/'epoch=15-step=847.ckpt' --gogoll_checkpoint_path /cluster/scratch/$USER/logs/whole_pipeline_test_1201-1709_34/gogoll/'epoch=73-step=3921.ckpt'
```

## Folder structure

Aside from the folder structure you'll get by cloning this branch, you will need to make sure your data folder has the following layout:

```
project
├── data
│   ├── exp
│   │   ├── rgb
│   │   └── semseg
│   └── other_domains
│       ├── domainA
│       └── domainB
├── configs
├── models
├── preprocessing
└── ...
```
