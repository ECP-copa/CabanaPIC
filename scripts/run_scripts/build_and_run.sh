#!/bin/bash

echo "Positional Parameters"
echo '$0 = ' $0 # this file
echo '$1 = ' $1 # repo path
echo '$2 = ' $2 # CXX
echo '$3 = ' $3 # kokkos install dir
echo '$4 = ' $4 # cabana install dir
echo '$5 = ' $5 # platform

cd $1 # CD into right folder
echo "--> Running $5 in $1 with $"

KOKKOS_INSTALL_DIR=$3
CABANA_INSTALL_DIR=$4
cxx=$2
platform=$5

options=""
if [[ $platform == "GPU" ]]; then
    options="-D ENABLE_GPU=ON"
    cxx="$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper"
elif [[ $platform == "Serial" ]]; then
    options="-D ENABLE_SERIAL=ON"
fi

mkdir build-$platform
cd build-$platform

# Build CPU *or* GPU?
# TODO: the way this selects the cmake folder is awful
 #-D CMAKE_CXX_COMPILER=$KOKKOS_SRC_DIR/bin/nvcc_wrapper \
CXX=$cxx cmake -DCMAKE_BUILD_TYPE=Release -DKOKKOS_DIR=$KOKKOS_INSTALL_DIR -DCABANA_DIR=$CABANA_INSTALL_DIR $options ../../../../..;
make VERBOSE=1

# Run the code and track the performance
{ time ./minipic > out ; } 2> time.txt
