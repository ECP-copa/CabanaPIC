#!/bin/bash

echo "Positional Parameters"
echo '$0 = ' $0 # this file
echo '$1 = ' $1 # cxx
echo '$2 = ' $2 # kokkos dir
echo '$3 = ' $3 # install dir
echo '$4 = ' $4 # platform
echo '$5 = ' $5 # arch

kokkos_dir=`pwd`/$2
install_dir=$3
platform=$4
cxx=$1

cd $kokkos_dir
mkdir $install_dir
cd $install_dir
echo `pwd`

options="--compiler=$cxx"

#platforms = ["Serial", "CPU", "GPU", "UVM"]
if [[ $platform == "Serial" ]]; then
    options="$options --with-serial"
elif [[ $platform == "CPU" ]]; then
    options="$options --with-openmp"
elif [[ $platform == "GPU" ]]; then
    #export NVCC_WRAPPER_DEFAULT_COMPILER=`which $CXX`
    options="--with-cuda --arch=Kepler30 --with-cuda-options=enable_lambda --compiler=$kokkos_dir/bin/nvcc_wrapper ;"
# TODO: enable UVM build
#elif [[ $platform == "UVM" ]] then
    #options="--"
else
    # This means they passed up the wrong value
    eep
fi

echo "Running with $options"

# TODO: check this works
CXX=$cxx ../generate_makefile.bash --prefix=`pwd`/install $options
make install
