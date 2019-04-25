#!/bin/bash

echo "Positional Parameters"
echo '$0 = ' $0 # this file
echo '$1 = ' $1 # cxx
echo '$2 = ' $2 # kokkos dir
echo '$3 = ' $3 # cabana dir
echo '$4 = ' $4 # install dir
echo '$5 = ' $5 # platform

cxx=$1
kokkos_dir=`pwd`/$2
echo $kokkos_dir
cabana_dir=$3
install_dir=$4
platform=$5

cd $cabana_dir
mkdir $install_dir
cd $install_dir
echo `pwd`

# Default to off
options="-D Cabana_ENABLE_Serial=OFF"

# possible platforms = ["Serial", "CPU", "GPU", "UVM"]

if [[ $platform == "Serial" ]]; then
    # Override default
    options="-D Cabana_ENABLE_Serial=ON"
elif [[ $platform == "CPU" ]]; then
    options="-D Cabana_ENABLE_OpenMP=ON $options"
elif [[ $platform == "GPU" ]]; then
    options="-D CMAKE_CXX_COMPILER=$kokkos_dir/bin/nvcc_wrapper -D Cabana_ENABLE_Cuda:BOOL=ON $options"
    cxx=$kokkos_dir/bin/nvcc_wrapper
# TODO: enable UVM build
#elif [[ $platform == "UVM" ]] then
    #options="--"
#else
    # This means they passed up the wrong value
    #eep
fi

echo $options

CXX=$cxx cmake \
     -D CMAKE_BUILD_TYPE="Release" \
     -D CMAKE_PREFIX_PATH=$kokkos_dir \
     -D CMAKE_INSTALL_PREFIX=`pwd`/install \
     -D Cabana_ENABLE_TESTING=OFF \
     -D Cabana_ENABLE_EXAMPLES=OFF \
     $options \
     .. ;
make install
