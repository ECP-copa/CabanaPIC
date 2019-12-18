#ifndef pic_helper_h
#define pic_helper_h

#include "logger.h"
#include "Cabana_ExecutionPolicy.hpp" // SIMDpolicy
#include "Cabana_Parallel.hpp" // Simd parallel for

#include "input/deck.h"

// Converts from an index that doesn't know about ghosts to one that does
//KOKKOS_INLINE_FUNCTION
int allow_for_ghosts(int pre_ghost)
{

    size_t ix, iy, iz;
    RANK_TO_INDEX(pre_ghost, ix, iy, iz,
            deck.nx,
            deck.ny);
    //    printf("%ld\n",ix);
    int with_ghost = VOXEL(ix, iy, iz,
            deck.nx,
            deck.ny,
            deck.nz,
            deck.num_ghosts);

    return with_ghost;
}

// Function to print out the data for every particle.
void print_particles( const particle_list_t particles )
{
    auto position_x = Cabana::slice<PositionX>(particles);
    auto position_y = Cabana::slice<PositionY>(particles);
    auto position_z = Cabana::slice<PositionZ>(particles);

    auto velocity_x = Cabana::slice<VelocityX>(particles);
    auto velocity_y = Cabana::slice<VelocityY>(particles);
    auto velocity_z = Cabana::slice<VelocityZ>(particles);

    auto weight = Cabana::slice<Weight>(particles);
    auto cell = Cabana::slice<Cell_Index>(particles);

    auto _print =
        KOKKOS_LAMBDA( const int s, const int i )
        {
                printf("Struct id %d offset %d \n", s, i);
                printf("Position x %e y %e z %e \n", position_x.access(s,i), position_y.access(s,i), position_z.access(s,i) );
        };

    // TODO: How much sense does printing in parallel make???
    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );

    //logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    //logger << "particles.numSoA() " << particles.numSoA() << std::endl;

    Cabana::simd_parallel_for( vec_policy, _print, "_print()" );

    std::cout << std::endl;

}

#endif // pic_helper_h
