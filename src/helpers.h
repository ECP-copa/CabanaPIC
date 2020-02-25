#ifndef pic_helper_h
#define pic_helper_h

#include "logger.h"
#include "Cabana_ExecutionPolicy.hpp" // SIMDpolicy
#include "Cabana_Parallel.hpp" // Simd parallel for

// Converts from an index that doesn't know about ghosts to one that does
KOKKOS_INLINE_FUNCTION int allow_for_ghosts(int pre_ghost)
{

    size_t ix, iy, iz;
    RANK_TO_INDEX(pre_ghost, ix, iy, iz,
            Parameters::instance().nx,
            Parameters::instance().ny);
    //    printf("%ld\n",ix);
    int with_ghost = VOXEL(ix, iy, iz,
            Parameters::instance().nx,
            Parameters::instance().ny,
            Parameters::instance().nz,
            Parameters::instance().num_ghosts);

    return with_ghost;
}

// Function to print out the data for every particle.
void print_particles( FILE * fp, const particle_list_t particles, const real_t xmin, const real_t ymin, const real_t zmin, const real_t dx, const real_t dy, const real_t dz,size_t nx,size_t ny,size_t nz, size_t ng)
{
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto weight = particles.slice<Weight>();
    auto cell = particles.slice<Cell_Index>();

    auto _print =
        KOKKOS_LAMBDA( const int s, const int i )
        {
	  //                printf("Struct id %d offset %d \n", s, i);
	  //                printf("%e  %e  %e \n", position_x.access(s,i), position_y.access(s,i), position_z.access(s,i) );
	  size_t ix,iy,iz;
	  int ii = cell.access(s, i);
	  RANK_TO_INDEX(ii, ix,iy,iz,nx+2*ng,ny+2*ng);
	  real_t x = xmin + (ix-1+(position_x.access(s,i)+1.0)*0.5)*dx;
	  real_t y = ymin + (iy-1+(position_y.access(s,i)+1.0)*0.5)*dy;
	  real_t z = zmin + (iz-1+(position_z.access(s,i)+1.0)*0.5)*dz;
	  fprintf(fp, "%e  %e  %e %d %d %d \n", x,y,z,ix,iy,iz);

        };

    // TODO: How much sense does printing in parallel make???
    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );

    //logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    //logger << "particles.numSoA() " << particles.numSoA() << std::endl;

    Cabana::simd_parallel_for( vec_policy, _print, "_print()" );
    fprintf(fp, "\n\n");

    std::cout << std::endl;

}

#endif // pic_helper_h
