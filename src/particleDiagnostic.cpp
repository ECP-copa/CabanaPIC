#include <cstddef> // size_t
#include <fstream>
#include <iostream>

#include "particleDiagnostic.h"
#include "types.h"

// collect particle moments in each cell
void particle_moment( particle_list_t &particles_i,
                      const real_t m_i,
                      size_t nx,
                      size_t ny,
                      size_t nz,		      
                      size_t ng,
                      const std::vector<size_t> &npc_scan_i,
                      const moment_array_t &moment_i,
                      const int debug )
{
    //  std::cout<<"particle_moments\n";
    auto vx = Cabana::slice<VelocityX>( particles_i );
    auto vy = Cabana::slice<VelocityY>( particles_i );
    auto vz = Cabana::slice<VelocityZ>( particles_i );
    auto wp = Cabana::slice<Weight>( particles_i );

    for ( size_t ix = 0; ix < nx; ++ix ) {
	for ( size_t iy = 0; iy < ny; ++iy ) {
	    for ( size_t iz = 0; iz < nz; ++iz ) {
		size_t ci  = VOXEL( ix, iy, iz, nx, ny, nz, ng ); 
		size_t Np_i = npc_scan_i[ci + 1] - npc_scan_i[ci];
		//std::cout<<ci<<","<<Np_i<<",";
		if ( Np_i == 0 ) {
		    continue; // nothing to do
		}
		mom_type T_i;
		T_i.value[0] = 0.0;
		T_i.value[1] = 0.0;
		T_i.value[2] = 0.0;
		T_i.value[3] = 0.0;
		T_i.value[4] = 0.0;
		T_i.value[5] = 0.0;
		T_i.value[6] = 0.0;
		Kokkos::RangePolicy<ExecutionSpace> reduce_policy( npc_scan_i[ci], npc_scan_i[ci + 1] );
		Kokkos::parallel_reduce(
					reduce_policy,
					moms<typename particle_list_t::template member_slice_type<VelocityX>>( vx, vy, vz, wp ),
					T_i );

		Kokkos::fence();
		moment_i( ci, 0 ) = T_i.value[0]; // total weight
		moment_i( ci, 1 ) = T_i.value[1] * m_i;
		moment_i( ci, 2 ) = T_i.value[2] * m_i;
		moment_i( ci, 3 ) = T_i.value[3] * m_i;
		moment_i( ci, 4 ) = T_i.value[4] * m_i;
		moment_i( ci, 5 ) = T_i.value[5] * m_i;
		moment_i( ci, 6 ) = T_i.value[6] * m_i;

		T_i.value[1] = T_i.value[1] / T_i.value[0];
		T_i.value[2] = T_i.value[2] / T_i.value[0];
		T_i.value[3] = T_i.value[3] / T_i.value[0];
		T_i.value[4] = T_i.value[4] / T_i.value[0];
		T_i.value[5] = T_i.value[5] / T_i.value[0];
		T_i.value[6] = T_i.value[6] / T_i.value[0];

		T_i.value[4] = ( T_i.value[4] - T_i.value[1] * T_i.value[1] ) * m_i;
		T_i.value[5] = ( T_i.value[5] - T_i.value[2] * T_i.value[2] ) * m_i;
		T_i.value[6] = ( T_i.value[6] - T_i.value[3] * T_i.value[3] ) * m_i;

		//	std::cout<<moment_i( ci, 0 )<<", "<<moment_i( ci, 1 )<<", "<<moment_i( ci, 6 )<<",";
		// if ( ci == 1 ) 
		//     std::cout << ci<<","<<T_i.value[0] << " " << T_i.value[1] << " " << T_i.value[2] << " "
		// 	      << T_i.value[3] << " " << T_i.value[4] << " " << T_i.value[5] << " "
		// 	      << T_i.value[6] << " " << ( T_i.value[4] + T_i.value[5] + T_i.value[6] ) / 3.0
		// 	      << "\n";
		
	    }
	}
    }
}

void print_2pcle( size_t step,
                  real_t dt,
                  particle_list_t &particles,
                  size_t nx,
                  size_t ny,
                  size_t ng,
                  real_t xmin,
                  real_t dx )
{
    auto position_x = Cabana::slice<PositionX>( particles );
    auto velocity_x = Cabana::slice<VelocityX>( particles );
    auto cell       = Cabana::slice<Cell_Index>( particles );
    auto _print     = KOKKOS_LAMBDA( const int s, const int i )
    {
        size_t pi = (s) *particle_list_t::vector_length + i;
        if ( pi == 0 ) {
            size_t ix, iy, iz;
            size_t ii = cell( pi );
            RANK_TO_INDEX( ii, ix, iy, iz, nx + 2 * ng, ny + 2 * ng );
            real_t x0 = xmin + ( ix - ng + ( position_x( pi ) + 1.0 ) * 0.5 ) * dx;
            iy        = iy == 1 ? iy : 1; // 1D only
            iz        = iz == 1 ? iz : 1;
            printf( "%e %e %e %lu %lu %lu %lu\n",
                    step * dt,
                    x0,
                    velocity_x( pi ),
                    (unsigned long) ix,
                    (unsigned long) iy,
                    (unsigned long) iz,
                    (unsigned long) ii );
        }
    };

    Cabana::SimdPolicy<particle_list_t::vector_length, ExecutionSpace> vec_policy(
        0, particles.size() );
    Cabana::simd_parallel_for( vec_policy, _print, "print()" );
}
