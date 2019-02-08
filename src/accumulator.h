#ifndef ACCUMULATOR_T
#define ACCUMULATOR_T

#include "grid.h"

class accumulator_t {
    public:
      float jx[4];   // jx0@(0,-1,-1),jx1@(0,1,-1),jx2@(0,-1,1),jx3@(0,1,1)
      float jy[4];   // jy0@(-1,0,-1),jy1@(-1,0,1),jy2@(1,0,-1),jy3@(1,0,1)
      float jz[4];   // jz0@(-1,-1,0),jz1@(1,-1,0),jz2@(-1,1,0),jz3@(1,1,0)

      accumulator_t() :
          jx { 0.0f, 0.0f, 0.0f, 0.f },
          jy { 0.0f, 0.0f, 0.0f, 0.f },
          jz { 0.0f, 0.0f, 0.0f, 0.f }
      {
          // empty
      }
};

// TODO: this can be replaced by a view
class accumulator_array_t {
    public:
        accumulator_t* a;
        size_t size;
        //int n_pipeline; // Number of pipelines supported by this accumulator
        //int stride;     // Stride be each pipeline's accumulator array
        //grid_t* g;
        accumulator_array_t(size_t size)
        {
            this->size = size;
            // TODO: obviously this will fail on non CPU, but works as a place holder
            // TODO: NEW
            a = (accumulator_t*)malloc( sizeof(accumulator_t) * size );
        }
};

#endif // header guard
