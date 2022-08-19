#CabanaPIC

## Installation

This code has two major dependencies:

1. Kokkos
2. Cabana

Instructions on how to obtain and install both can be found [here](https://github.com/ECP-copa/Cabana/wiki/Build-Instructions)

Once these are installed, you can configure and build this project using CMake.
The only necessary configuration argument is the path to Cabana (which will
also bring in Kokkos). An example build line will look something like this:

```
cmake -DCMAKE_PREFIX_PATH="$HOME/Cabana/build/install" ..
```

CabanaPIC uses the default enabled Kokkos backend (see more information
[here](https://github.com/kokkos/kokkos/wiki/Initialization#51-initialization-by-command-line-arguments)).
It is possible to require a CPU build by adding `-DREQUIRE_HOST=ON` (which uses
the default enabled host backend).

The default field solver is "EM"; to use the "ES" solver, add `-DSOLVER_TYPE="ES"`.


## Running

Users can compile in custom input decks by specifying `INPUT_DECK` at build
time, e.g:

```
cmake -DCMAKE_PREFIX_PATH="$HOME/Cabana/build/install" -DINPUT_DECK=./decks/2stream-short.cxx ..
```

Some example decks live in `./decks`. Custom decks must follow the layout put
forth in `./src/input/decks.h`

## Feature Wishlist

1. Configurable to run in different precisions (real_t to configure float/double)
2. The particle data store layout should be configurable (AoS/SoA/AoSoA)
3. The particle shape function used should be configurable

## Copyright

© (or copyright) 2019. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

This is open source software; you can redistribute it and/or modify it under the terms of the BSD-3 License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
