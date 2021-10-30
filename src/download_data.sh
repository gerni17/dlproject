#!/bin/bash

# Download dataset 
if [ ! -d "../data/exp" ]; then
  echo "Download exp"
  wget 'https://polybox.ethz.ch/index.php/s/yEAssV8NCli0UbW/download' -P ../data/
  echo "Extract exp data"
  unzip ../data/exp.zip -d ../data/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm ../data/exp.zip
  echo "\n"
fi

if [ ! -d "../data/domain_shift" ]; then
  echo "Download domain_shift dataset"
  wget 'https://polybox.ethz.ch/index.php/s/l3QUimOE1Jw8830/download' -P ../data/
  echo "Extract domain_shift data"
  unzip ../data/SegData.zip -d ../data/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm ../data/domain_shift.zip
  echo "\n"
fi

# conda create -n pip_torch
# conda activate pip_torch
# pip install -r requirements.txt
