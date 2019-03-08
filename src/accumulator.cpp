// TODO: add namespace?

#include "accumulator.h"

void unload_accumulator_array(
        field_array_t fields,
        accumulator_array_t accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz
)
{

    auto jfx = fields.slice<FIELD_JFX>();
    auto jfy = fields.slice<FIELD_JFY>();
    auto jfz = fields.slice<FIELD_JFZ>();

    // TODO: give these real values
    float cx = 1.0f; // 0.25 * fa->g->rdy * fa->g->rdz / fa->g->dt;
    float cy = 1.0f; // 0.25 * fa->g->rdz * fa->g->rdx / fa->g->dt;
    float cz = 1.0f; // 0.25 * fa->g->rdx * fa->g->rdy / fa->g->dt;

    // This is a hang over for VPIC's nasty type punting
    const size_t JX_OFFSET = 0;
    const size_t JY_OFFSET = 4;
    const size_t JZ_OFFSET = 8;

    size_t x_offset = 1; // VOXEL(x+1,y,  z,   nx,ny,nz);
    size_t y_offset = (1*nx); // VOXEL(x,  y+1,z,   nx,ny,nz);
    size_t z_offset = (1*nx*ny); // VOXEL(x,  y,  z+1, nx,ny,nz);

    auto a0 = accumulators.slice<0>();

    // TODO: we have to be careful we don't reach past the ghosts here
    auto _unload_accumulator = KOKKOS_LAMBDA( const int i )
    {
        // Original:
        // f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );

        jfx(i) += cx*(
                    a0(i,JX_OFFSET+0) +
                    a0(i+y_offset,JX_OFFSET+1) +
                    a0(i+z_offset,JX_OFFSET+2) +
                    a0(i+y_offset+z_offset,JX_OFFSET+3)
                );

        jfy(i) += cy*(
                    a0(i,JY_OFFSET+0) +
                    a0(i+z_offset,JY_OFFSET+1) +
                    a0(i+y_offset,JY_OFFSET+2) +
                    a0(i+y_offset+z_offset,JY_OFFSET+3)
                );

        jfz(i) += cz*(
                    a0(i,JZ_OFFSET+0) +
                    a0(i+x_offset,JZ_OFFSET+1) +
                    a0(i+y_offset,JZ_OFFSET+2) +
                    a0(i+x_offset+y_offset,JZ_OFFSET+3)
                );
    };

    /* // Crib sheet for old variable names
    a0  = &a(x,  y,  z  );
    ax  = &a(x-1,y,  z  );
    ay  = &a(x,  y-1,z  );
    az  = &a(x,  y,  z-1);
    ayz = &a(x,  y-1,z-1);
    azx = &a(x-1,y,  z-1);
    axy = &a(x-1,y-1,z  )
    */

}
