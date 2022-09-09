#ifndef PARTICLE_SORT_H
#define PARTICLE_SORT_H

#include <cstddef> // size_t
#include <iostream>

#include "types.h"

void particle_sort( particle_list_t &particles,
                    size_t nx,
		    size_t ny,
		    size_t nz,
		    size_t ng,
                    KokkosRngPool &rand_pool,
                    std::vector<size_t> &npc_scan,
                    size_t nip,
		    size_t np0=0);

//without shuffle
void particle_sort(particle_list_t &particles,
                    size_t nx,
		    size_t ny,
		    size_t nz,
		    size_t ng,                    
                    std::vector<size_t> &npc_scan,
                    const size_t Np,
		    const size_t np0);

#endif
