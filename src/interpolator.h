#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <cstdint>
#include <cstddef>

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include "types.h"
#include "fields.h"

/**
 * @brief Class to store interpolator data for use in kernels
 */
// TODO: This will have to change based on the order of the particle function
class interpolator_t {

    public:
        float ex, dexdy, dexdz, d2exdydz;
        float ey, deydz, deydx, d2eydzdx;
        float ez, dezdx, dezdy, d2ezdxdy;
        float cbx, dcbxdx;
        float cby, dcbydy;
        float cbz, dcbzdz;

        interpolator_t() :
            ex(0.0), dexdy(0.0), dexdz(0.0), d2exdydz(0.0),
            ey(0.0), deydz(0.0), deydx(0.0), d2eydzdx(0.0),
            ez(0.0), dezdx(0.0), dezdy(0.0), d2ezdxdy(0.0),
            cbx(0.0), dcbxdx(0.0),
            cby(0.0), dcbydy(0.0),
            cbz(0.0), dcbzdz(0.0) { }

        // TODO: make sure the padding is done during allocation
        //float _pad[2];  // 16-byte align
};

void load_interpolator_array(
        field_array_t fields,
        interpolator_array_t interpolators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz
);

#endif
