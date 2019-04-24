#!/bin/bash

echo "Positional Parameters"
echo '$0 = ' $0 # this file
echo '$1 = ' $1 # repo path
echo '$2 = ' $2 # CXX
echo '$3 = ' $3 # kokkos install dir
echo '$4 = ' $4 # cabana install dir

cd $1 # CD into right folder

KOKKOS_INSTALL_DIR=$3
CABANA_INSTALL_DIR=$4
CXX=$2

mkdir build
cd build

# Build CPU *or* GPU?
 #-D CMAKE_CXX_COMPILER=$KOKKOS_SRC_DIR/bin/nvcc_wrapper \
CXX=$CXX cmake -DKOKKOS_DIR=$KOKKOS_INSTALL_DIR -DCABANA_DIR=$CABANA_INSTALL_DIR ..;
make VERBOSE=1

# Run the code and track the performance
{ time ./minipic > out ; } 2> time.txt
