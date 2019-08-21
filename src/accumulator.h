#ifndef ACCUMULATOR_T
#define ACCUMULATOR_T

#include <cstdint>
#include <cstddef>

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include "types.h"
#include "grid.h"
#include "fields.h"

/*
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
*/

void clear_accumulator_array(
        field_array_t& fields,
        accumulator_array_t& accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz
);

void unload_accumulator_array(
        field_array_t& fields,
        accumulator_array_t& accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz,
        size_t ng,
        real_t dx,
        real_t dy,
        real_t dz,
        real_t dt
);

#endif // header guard
