#ifndef BOUNDARY_P_H
#define BOUNDARY_P_H

#include "types.h"

void boundary_p(
        particle_list_t& particles,
        particle_list_t& particles_copy,
        int max_nm) // TODO: can we tighten this bound to nm?
{

    // Make sure movers and particle_copy are accessible in the space they're needed

    // Build a list of ranks the particles have to move to

    // Call to Cabana distributor
    // TODO: this does not need to be realloced every step
    particle_ranklist_t particle_comm("particle_comm", max_nm);

    const int Comm_Rank = 0;

    auto particle_exports = Cabana::slice<Comm_Rank>(particle_comm);

    auto _find_ranks =
        KOKKOS_LAMBDA( const int s, const int i )
        {
            particle_exports.access(s, i) = -1;
        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particle_comm.size() );
    Cabana::simd_parallel_for( vec_policy, _find_ranks, "find_particle_ranks()" );

    auto particle_distributor = Cabana::Distributor<MemorySpace>(
            grid_comm, particle_exports );

    // TODO: Think we need to set up neighbors
    // Cabana::migrate( particle_distributor, particles_copy );
}

#endif // BOUNDARY_P_H
