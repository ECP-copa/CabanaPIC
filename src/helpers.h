#ifndef pic_helper_h
#define pic_helper_h

#include "logger.h"

// TODO: this may be a bad name?
# define RANK_TO_INDEX(rank,ix,iy,iz,_x,_y) \
    int _ix, _iy, _iz;                                                    \
    _ix  = (rank);                        /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(_x);   /* iy = iy+gpy*iz */            \
    _ix -= _iy*int(_x);   /* ix = ix */                   \
    _iz  = _iy/int(_y);   /* iz = iz */                   \
    _iy -= _iz*int(_y);   /* iy = iy */                   \
    (ix) = _ix;                                                           \
    (iy) = _iy;                                                           \
    (iz) = _iz;                                                           \

#define VOXEL(x,y,z, nx,ny,nz, NG) ((x) + ((nx)+(NG*2))*((y) + ((ny)+(NG*2))*(z)))

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
void print_particles( const particle_list_t particles )
{
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto charge = particles.slice<Charge>();
    auto cell = particles.slice<Cell_Index>();

    auto _print =
        KOKKOS_LAMBDA( const int s, const int i )
        {
                printf("Struct id %d offset %d \n", s, i);
                printf("Position x %e y %e z %e \n", position_x.access(s,i), position_y.access(s,i), position_z.access(s,i) );

                //std::cout << "Velocity "
                    //<< velocity_x.access(s,i) << " "
                    //<< velocity_y.access(s,i) << " "
                    //<< velocity_z.access(s,i) << " "
                    //<< ". Cell: " << cell.access(s,i);
                //std::cout << std::endl;
        };

    // TODO: How much sense does printing in parallel make???
    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );
    logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    Cabana::simd_parallel_for( vec_policy, _print, "_print()" );

    std::cout << std::endl;

}

#endif // pic_helper_h
