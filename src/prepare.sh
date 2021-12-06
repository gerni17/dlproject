#!/bin/bash

# not necessary to run
echo "Start prepare"
cd /home/gian/git/deeplabv3p/src
python3 -m mtl.datasets.prepare
