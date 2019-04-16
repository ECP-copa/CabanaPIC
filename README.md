# Overview 

This document is to be used to collect a wish list of features before
development starts

## Installation

This code has two major dependencies:

1. Kokkos
2. Cabana

Instructions on how to obtain and install both can be found [here](https://github.com/ECP-copa/Cabana/wiki/Build-Instructions)

Once these are installed, you can configurable and build this project using
cmake. To do so, the user should manually set the following two variables:

1. `CABANA_DIR` (pointing to the install location of Cabana)
2. `KOKKOS_DIR` (pointing to the install location of Kokkos)

An example build line will look something like this:

```
cmake -DKOKKOS_DIR=$HOME/tools/kokkos -DCABANA_DIR=$HOME/tools/cabana ..
```

For GPU builds, you additionally need to point the CXX compiler to the Kokkos
Cuda wrapper, you can do this by doing something like:

```
CXX=$HOME/tools/kokkos/bin/nvcc_wrapper cmake -DKOKKOS_DIR=$HOME/tools/kokkos -DCABANA_DIR=$HOME/tools/cabana ..
```

## Feature Wishlist

1. Configurable to run in different precisions (real_t to configure float/double)
2. The particle data store layout should be Configurable (AoS/SoA/AoSoA)
3. The particle shape function used should be configurable 

## Copyright

© (or copyright) 2019. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

This is open source software; you can redistribute it and/or modify it under the terms of the BSD-3 License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
