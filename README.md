# Deep Learning Project (Aleksandar Milojevic, Mugeeb Hassan, Gian Erni)

## Getting started

We provide information to reproduce our results on Euler (new software stack).

### To run the pipeline on euler run the following commands:

Clone the repo:

```sh
git clone git@gitlab.ethz.ch:dlproject/dlproject.git
```

Download and run the files:

```sh
bsub -n 2 -W 72:00 -R "rusage[scratch=1000,mem=15000,ngpus_excl_p=1]" -oo /cluster/scratch/$USER/log < run.sh
```

To add commandline arguments you can add them in the bash file run.sh, all the possible command line files can be found in the files inside the config folder.
