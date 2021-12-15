# How to use: download.sh $(root_dir)

root_dir=${1:-/cluster/scratch/$USER/dl_data}
data_dir=$root_dir"/data/"
echo "Downloading dataset into $root_dir"
echo "what $data_dir"

if [ ! -d "$data_dir/source" ]; then
  echo "Download source"
  wget 'https://polybox.ethz.ch/index.php/s/Q1A7Yytt2vOvMDy/download' -P $root_dir/data/
  mv $data_dir/download $data_dir/source.zip
  echo "Extract source data"
  unzip $data_dir/source.zip -d $data_dir | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm $data_dir/source.zip
  echo "\n"
fi

if [ ! -d "$data_dir/medium" ]; then
  echo "Download medium dataset"
  wget 'https://polybox.ethz.ch/index.php/s/ZqCQM58gcnrdQB9/download' -P $data_dir
  mv $data_dir/download $data_dir/medium.zip
  echo "Extract medium data"
  unzip $data_dir/medium.zip -d $data_dir | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm $data_dir/medium.zip
  echo "\n"
fi

if [ ! -d "$data_dir/easy" ]; then
  echo "Download easy dataset"
  wget 'https://polybox.ethz.ch/index.php/s/CgHgBQ97C0DIuXs/download' -P $data_dir/
  mv $data_dir/download $data_dir/easy.zip
  echo "Extract easy data"
  unzip $data_dir/easy.zip -d $data_dir/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm $data_dir/easy.zip
  echo "\n"
fi
