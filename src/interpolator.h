#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

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

/**
 * @brief Class to hold a pointer to the interpolator array data
 */
class interpolator_array_t {
    public:
        interpolator_t* i;
        size_t size;
        interpolator_array_t(size_t size)
        {
            this->size = size;
            // TODO: obviously this will fail on non CPU, but works as a place holder
            i = (interpolator_t*)malloc( sizeof(interpolator_t) * size );
        }
        //grid_t * g; // TODO: Do we need a grid here? It's just a holder for some metadata
};

#endif
