# Deep Learning Project

## Getting started

Make sure you install all required packages first (e.g wandb, pytorch, pytorch lightning etc)

To run the cyclegan experiment run the following command. If yo

```sh
python -m scripts.train_cyclegan --name cyclegan_experiment --log_dir ./logs --dataset_root ./data --num_epochs 8 --use_wandb True
```

To see which command line options you have available you can call

```sh
python -m scripts.train_cyclegan --help
```

## Folder structure

Aside from the folder structure you'll get by cloning this branch, you will need to make sure your data folder has the following layout:
project
├── data
│ ├── exp
│ │ ├── test
│ │ │ ├── rgb
│ │ │ └── semseg
│ │ ├── train
│ │ │ ├── rgb
│ │ │ └── semseg
│ │ └── val
│ │ ├── rgb
│ │ └── semseg
│ └── other_domains
│ ├── test
│ │ ├── domainA
│ │ └── domainB
│ └── train
│ ├── domainA
│ └── domainB
├── configs
├── models
├── preprocessing
└── ...
