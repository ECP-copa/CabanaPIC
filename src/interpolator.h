#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <cstdint>
#include <cstddef>

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include "types.h"
#include "fields.h"

void load_interpolator_array(
        field_array_t fields,
        interpolator_array_t interpolators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz,
        size_t ng
);

void initialize_interpolator(interpolator_array_t& f0);

#endif
