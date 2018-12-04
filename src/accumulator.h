#ifndef ACCUMULATOR_T
#define ACCUMULATOR_T

#include "grid.h"

typedef struct accumulator {
  float jx[4];   // jx0@(0,-1,-1),jx1@(0,1,-1),jx2@(0,-1,1),jx3@(0,1,1)
  float jy[4];   // jy0@(-1,0,-1),jy1@(-1,0,1),jy2@(1,0,-1),jy3@(1,0,1)
  float jz[4];   // jz0@(-1,-1,0),jz1@(1,-1,0),jz2@(-1,1,0),jz3@(1,1,0)
} accumulator_t;

typedef struct accumulator_array {
  accumulator_t* a;
  int n_pipeline; // Number of pipelines supported by this accumulator
  int stride;     // Stride be each pipeline's accumulator array
  grid_t* g;
} accumulator_array_t;

#endif // header guard
