#!/bin/bash

echo "Positional Parameters"
echo '$0 = ' $0 # this file
echo '$1 = ' $1 # kokkos dir
echo '$2 = ' $2 # install dir
echo '$3 = ' $3 # platforms
echo '$4 = ' $4 # arch

kokkos_dir=$1
install_dir=$2
platform=$3

cd $kokkos_dir
mkdir $install_dir
cd $install_dir
echo `pwd`


options=""

#platforms = ["Serial", "CPU", "GPU", "UVM"]
if [[ $platform == "Serial" ]]; then
    options="--with-serial"
elif [[ $platform == "CPU" ]]; then
    options="--with-openmp"
elif [[ $platform == "GPU" ]]; then
    options="--with-cuda --arch=Kepler30 --with-cuda-options=enable_lambda --compiler=$kokkos_dir/bin/nvcc_wrapper ;"
# TODO: enable UVM build
#elif [[ $platform == "UVM" ]] then
    #options="--"
else
    # This means they passed up the wrong value
    eep
fi

../generate_makefile.bash \
  --prefix=$install_dir/install $options;
make install
