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

## Feature Wishlist

1. Configurable to run in different precisions (real_t to configure float/double)
2. The particle data store layout should be Configurable (AoS/SoA/AoSoA)
3. The particle shape function used should be configurable 
