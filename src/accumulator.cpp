// TODO: add namespace?

#include "accumulator.h"

void clear_accumulator_array(
        field_array_t& fields,
        accumulator_array_t& accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz
)
{
    auto _clean_accumulator = KOKKOS_LAMBDA(const int i)
    {
        /*
           a0(i,JX_OFFSET+0) = 0;
           a0(i+y_offset,JX_OFFSET+1) = 0;
           a0(i+z_offset,JX_OFFSET+2) = 0;
           a0(i+y_offset+z_offset,JX_OFFSET+3) = 0;

           a0(i,JY_OFFSET+0) = 0;
           a0(i+z_offset,JY_OFFSET+1) = 0;
           a0(i+y_offset,JY_OFFSET+2) = 0;
           a0(i+y_offset+z_offset,JY_OFFSET+3) = 0;

           a0(i,JZ_OFFSET+0) = 0;
           a0(i+x_offset,JZ_OFFSET+1) = 0;
           a0(i+y_offset,JZ_OFFSET+2) = 0;
           a0(i+x_offset+y_offset,JZ_OFFSET+3) = 0;
         */

      for (int j = 0; j < ACCUMULATOR_VAR_COUNT; j++)
      {
          for (int k = 0; k < ACCUMULATOR_VAR_COUNT; k++)
          {
              accumulators(i, j, k) = 0.0;
          }
      }
    };

    Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
    Kokkos::parallel_for( exec_policy, _clean_accumulator, "clean_accumulator()" );
}


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
)
{

    auto jfx = fields.slice<FIELD_JFX>();
    auto jfy = fields.slice<FIELD_JFY>();
    auto jfz = fields.slice<FIELD_JFZ>();

    // TODO: give these real values
    real_t cx = 0.25 / (dy * dz * dt);
    real_t cy = 0.25 / (dz * dx * dt);
    real_t cz = 0.25 / (dx * dy * dt);

    // This is a hang over for VPIC's nasty type punting
    const size_t JX_OFFSET = 0;
    const size_t JY_OFFSET = 4;
    const size_t JZ_OFFSET = 8;

    size_t x_offset = 1; // VOXEL(x+1,y,  z,   nx,ny,nz);
    size_t y_offset = (1*nx); // VOXEL(x,  y+1,z,   nx,ny,nz);
    size_t z_offset = (1*nx*ny); // VOXEL(x,  y,  z+1, nx,ny,nz);

    // TODO: we have to be careful we don't reach past the ghosts here
    auto _unload_accumulator = KOKKOS_LAMBDA( const int x, const int y, const int z )
    {
        // Original:
        // f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );
        int i = VOXEL(x,y,z, nx,ny,nz,ng);

        jfx(i) = cx*(
                    accumulators(i,                   accumulator_var::jx, 0) +
                    accumulators(i+y_offset,          accumulator_var::jx, 1) +
                    accumulators(i+z_offset,          accumulator_var::jx, 2) +
                    accumulators(i+y_offset+z_offset, accumulator_var::jx, 3)
                );

        jfy(i) = cy*(
                    accumulators(i,                   accumulator_var::jy, 0) +
                    accumulators(i+z_offset,          accumulator_var::jy, 1) +
                    accumulators(i+y_offset,          accumulator_var::jy, 2) +
                    accumulators(i+y_offset+z_offset, accumulator_var::jy, 3)
                );

        jfz(i) = cz*(
                    accumulators(i,                   accumulator_var::jz, 0) +
                    accumulators(i+x_offset,          accumulator_var::jz, 1) +
                    accumulators(i+y_offset,          accumulator_var::jz, 2) +
                    accumulators(i+x_offset+y_offset, accumulator_var::jz, 3)
                );
    };

    //std::cout << "Looping from " << ng << " to " << nx+ng << std::endl;
    Kokkos::MDRangePolicy< Kokkos::Rank<3> > non_ghost_policy( {ng,ng,ng}, {nx+ng, ny+ng, nz+ng} ); // Try not to into ghosts // TODO: dry this
    Kokkos::parallel_for( non_ghost_policy, _unload_accumulator, "unload_accumulator()" );

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
