###
# OSX has no wget command
# brew install wget
###

wget http://cbcl.mit.edu/projects/cbcl/software-datasets/pedestrians128x64.tar.gz
tar -xvzf pedestrians128x64.tar.gz
rm pedestrians128x64.tar.gz
mv -T pedestrians128x64 humans

wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip PennFudanPed.zip
rm PennFudanPed.zip

mkdir not-humans
julia preprocess_data.jl

wget http://juliaimages.github.io/ImageFeatures.jl/latest/img/humans.jpg

