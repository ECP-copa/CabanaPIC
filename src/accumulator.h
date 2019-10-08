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
