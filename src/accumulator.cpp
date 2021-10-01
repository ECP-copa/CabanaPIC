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
          for (int k = 0; k < ACCUMULATOR_ARRAY_LENGTH; k++)
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

    auto jfx = Cabana::slice<FIELD_JFX>(fields);
    auto jfy = Cabana::slice<FIELD_JFY>(fields);
    auto jfz = Cabana::slice<FIELD_JFZ>(fields);

    // TODO: give these real values
    //    printf("cx %e dy %e dz %e dt %e \n", dy, dz, dt);
    //real_t cx = 0.25 * (1.0 / (dy * dz)) / dt;
    real_t cx = 0.25 / (dy * dz * dt);
    real_t cy = 0.25 / (dz * dx * dt);
    real_t cz = 0.25 / (dx * dy * dt);

    // TODO: we have to be careful we don't reach past the ghosts here
    auto _unload_accumulator = KOKKOS_LAMBDA( const int x, const int y, const int z )
    {
        // Original:
        // f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );
        int i = VOXEL(x,y,z, nx,ny,nz,ng);

        // TODO: this level of re-calculation is overkill
        size_t x_down  = VOXEL(x-1, y,   z,   nx,ny,nz,ng);
        size_t y_down  = VOXEL(x,   y-1, z,   nx,ny,nz,ng);
        size_t z_down  = VOXEL(x,   y,   z-1, nx,ny,nz,ng);

        size_t xz_down = VOXEL(x-1, y,   z-1, nx,ny,nz,ng);
        size_t xy_down = VOXEL(x-1, y-1, z,   nx,ny,nz,ng);
        size_t yz_down = VOXEL(x,   y-1, z-1, nx,ny,nz,ng);

        jfx(i) = cx*(
                    accumulators(i,       accumulator_var::jx, 0) +
                    accumulators(y_down,  accumulator_var::jx, 1) +
                    accumulators(z_down,  accumulator_var::jx, 2) +
                    accumulators(yz_down, accumulator_var::jx, 3)
                );

        jfy(i) = cy*(
                    accumulators(i,       accumulator_var::jy, 0) +
                    accumulators(z_down,  accumulator_var::jy, 1) +
                    accumulators(x_down,  accumulator_var::jy, 2) +
                    accumulators(xz_down, accumulator_var::jy, 3)
                );

        jfz(i) = cz*(
                    accumulators(i,       accumulator_var::jz, 0) +
                    accumulators(x_down,  accumulator_var::jz, 1) +
                    accumulators(y_down,  accumulator_var::jz, 2) +
                    accumulators(xy_down, accumulator_var::jz, 3)
                );
    };

    //may not be enough if particles run into ghost cells
    Kokkos::MDRangePolicy< Kokkos::Rank<3> > non_ghost_policy( {ng,ng,ng}, {nx+ng+1, ny+ng+1, nz+ng+1} ); // Try not to into ghosts // TODO: dry this
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


void accumulate_rho_p_1D(
			 const particle_list_t & particles,
			 const rho_array_t & rho_accumulator,
			 size_t nx,
			 size_t ny,
			 size_t nz,
			 size_t ng,
			 real_t dx,
			 real_t dy,
			 real_t dz,
			 real_t qsp)
{
    auto position_x = Cabana::slice<PositionX>( particles );
    auto weight = Cabana::slice<Weight>( particles );
    auto cell   = Cabana::slice<Cell_Index>( particles );
    real_t cx = qsp / ( dy ); // 1D in y only

    for(int i=0; i<ny; ++i){
	const int f0 = VOXEL(1,   i+ng,   1,   nx, ny, nz, ng);
	//printf("%d %f\n",i,rho_accumulator(f0));
	std::cout<<rho_accumulator(f0)<<std::endl;
    }


    auto _collect_rho = KOKKOS_LAMBDA( const int s, const int i )
    {
	int ii  = cell.access( s, i );
	
	real_t wp = weight.access( s, i );
	rho_accumulator(ii) += wp*cx; //NGP, 1D
	/*
	real_t xp = position_x.access( s, i );
	rho_accumulator(ii) += (0.75 - xp*xp*0.25)*wp;
	rho_accumulator(ii+1) += (0.5*(0.5+xp*0.5)*(0.5+xp*0.5))*wp;
	rho_accumulator(ii-1) += (0.5*(0.5-xp*0.5)*(0.5-xp*0.5))*wp;
	*/
    };
    Cabana::SimdPolicy<particle_list_t::vector_length, Kokkos::DefaultHostExecutionSpace> vec_policy(0, particles.size() );
    Cabana::simd_parallel_for( vec_policy, _collect_rho, "collect_rho()" );

    /*
    for(int i=0; i<ny; ++i){
	const int f0 = VOXEL(1,   i+ng,   1,   nx, ny, nz, ng);
	std::cout<<(0.5+i)*dx<<" "<<rho_accumulator(f0)<<" "<<f0<<std::endl;;
    }

    exit(1);
    */
}
