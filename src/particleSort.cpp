
#include <iostream>
// kokkos rng
// save diagnostic state
#pragma GCC diagnostic push
// turn off the specific warning. Can also use "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"

// reset warnings
#pragma GCC diagnostic pop

#include "particleSort.h"


void particle_sort( particle_list_t &particles,
                    size_t nx,
		    size_t ny,
		    size_t nz,
		    size_t ng,
                    std::vector<size_t> &npc_scan,
                    const size_t Np,  //number of particles
		    const size_t np0) //starting point
{
    size_t ncell = (nx+2*ng)*(ny+2*ng)*(nz+2*ng);
    size_t num_bin = npc_scan.size()-2;
    assert(ncell<npc_scan.size());
    size_t ix      = 1; //allow ghost cells
    size_t iy      = 1;
    size_t iz      = 1;
    size_t ci_min  = VOXEL( ix, iy, iz, nx, ny, 1, ng );

    //    std::cout<<"ci_min = "<<ci_min<<", nx="<<nx<<", ny="<<ny<<std::endl;
    int b_min = 0, b_max = b_min+1; // relative to ci_min
    auto keys = Cabana::slice<Cell_Index>( particles );
    using SliceType   = decltype( keys );
    using DeviceType  = typename SliceType::device_type;
    using KeyViewType = Kokkos::View<typename SliceType::value_type *, DeviceType>;
    
    //     std::cout << "Particle sort, Np=" << Np<<", np0="<<np0<<std::endl;

    if ( ncell > 1 && Np > 0 ) {
	    // copy slice to keys
	KeyViewType kkeys( Kokkos::ViewAllocateWithoutInitializing( "k_slice_keys" ), Np );
	Kokkos::RangePolicy<typename DeviceType::execution_space> exec_policy( 0, Np );


	auto copy_op = KOKKOS_LAMBDA( const int i )
	    {
	     kkeys( i ) = keys( np0+i );
             size_t ix, iy, iz;
             size_t ii = kkeys(i);
             RANK_TO_INDEX( ii, ix, iy, iz, ( nx + ( 2 * ng ) ), ( ny + ( 2 * ng ) ) );

             //if(ix<1) printf("i=%zu,key=%zu,%zu,%zu,%zu\n",i,ix,iy,iz,ii);
            // if(ix>ncell) printf("i=%zu,key=%zu,%zu,%zu,%zu\n",i,ix,iy,iz,ii);
	     //if(i<10) printf("i=%d, np0+i=%d, kkeys=%d, keys=%d\n",i, np0+i, kkeys(i), keys(np0+i));
	    };
	Kokkos::parallel_for( "sortCopySliceToKeys_copy_op", exec_policy, copy_op );
	Kokkos::fence();
	if(Np==1){
	      int cell_id,b_id;
	      auto key = Kokkos::subview(kkeys, 0);
	      int scalar;
	      // Deep Copy Scalar View into a scalar
	      Kokkos::deep_copy(cell_id, key);
	      b_id = cell_id-ci_min;
	      //std::cout<<"cell_id="<<cell_id<<",ci_min="<<ci_min<<"\n";
	      num_bin = 1;
	      b_min = cell_id-ci_min+1;
	      b_max = b_min;
	      for ( size_t b = 0; b < b_min; ++b ) {
		  npc_scan[b] = 0;
	      }
	    // for ( size_t b = b_min; b < b_max; ++b ) {
	    // 	npc_scan[b] = 1;
	    // 	std::cout<<b<<" "<<npc_scan[b]<<std::endl;
	    // }
	}else {	
	      // find the bounds of keys
	      using KeyValueType = typename KeyViewType::non_const_value_type;
	      Kokkos::MinMaxScalar<KeyValueType> result;
	      Kokkos::MinMax<KeyValueType> reducer( result );
	      Kokkos::parallel_reduce(
				      "sortKeysMinMax",
				      Kokkos::RangePolicy<typename DeviceType::execution_space>( 0, Np ),
				      KOKKOS_LAMBDA( std::size_t i, decltype( result ) &local_minmax ) {
					  auto const val = kkeys( i );
					  if ( val < local_minmax.min_val ) {
					      local_minmax.min_val = val;
					      //printf("min: %zu %d\n",i,val);
					  }
					  if ( val > local_minmax.max_val ) {
					      local_minmax.max_val = val;
					      //printf("max: %zu %d\n",i,val);
					  }
				      },
				      reducer );
	      Kokkos::fence();

	      num_bin =
		  result.max_val - result.min_val; // ncell-1; //bin_data.numBin()=ncell, not sure why.
	      b_min = result.min_val - ci_min;
	      b_max = result.max_val - ci_min + 1;
	      // std::cout<<"result_min="<<result.min_val<<", ci_min="<<ci_min<<std::endl;
	      // std::cout<<"b_min="<<b_min<<", b_max="<<b_max<<", ncell="<<ncell<<", num_bin="<<num_bin<<std::endl;
	      if(b_min+1<b_max){
		  auto bin_data = Cabana::binByKey( keys, num_bin, np0, np0+Np );
		  Cabana::permute( bin_data, particles );
		  for ( size_t b = b_min; b < b_max; ++b ) {
		      npc_scan[b] = bin_data.binOffset( b - b_min ); // May be on different spaces
		      //std::cout<<b<<" "<<npc_scan[b]<<std::endl;
		  }
	      }else{
		  npc_scan[b_min]=0;
	      }
	      // std::cout <<"max="<<result.max_val<<", min="<<result.min_val<<std::endl;
	      // std::cout << "bin_data.numBin() = " << bin_data.numBin() << std::endl;
	      // std::cout << "ncell = "<< ncell <<", num_bin="<<num_bin<<std::endl;

	      // assert (ncell==size_t(bin_data.numBin()));
	      for ( size_t b = 0; b < b_min; ++b ) {
		  npc_scan[b] = 0;
	      }
	}//Np>1
    } else {
        for ( size_t b = 0; b < ncell + 1; ++b ) {
            npc_scan[b] = 0;
        }

    }
    for ( size_t b = b_max; b < ncell + 1; ++b ) {
        npc_scan[b] = Np;
        // std::cout<<b<<" "<<npc_scan[b]<<std::endl;
    }
}


void particle_sort( particle_list_t &particles,
                    size_t nx,
		    size_t ny,
		    size_t nz,
		    size_t ng,
                    KokkosRngPool &rand_pool,
                    std::vector<size_t> &npc_scan,
                    size_t nip,
		    size_t np0)
{
    size_t ncell = (nx+2*ng)*(ny+2*ng)*(nz+2*ng);    
    size_t Np      = particles.size() - nip;
    particle_sort(particles, nx, ny, nz, ng, npc_scan, Np, np0);
    // Kokkos random number generator
    using GeneratorType = KokkosRngPool::generator_type;

    Kokkos::View<real_t *> keys_b( "particle_shuffle_key", Np );
    // GeneratorPool rand_pool(5374858);

    auto _init_shuffle_keys = KOKKOS_LAMBDA( const int s, const int i )
    {
        size_t ip              = (s) *particle_list_t::vector_length + i;
        GeneratorType rand_gen = rand_pool.get_state();
        real_t x               = rand_gen.drand( 1.0 );
        keys_b( ip )           = x;
        rand_pool.free_state( rand_gen );
    };

    Cabana::SimdPolicy<particle_list_t::vector_length, ExecutionSpace> vec_policy( 0, Np );
    Cabana::simd_parallel_for( vec_policy, _init_shuffle_keys, "init_shuffle_keys_particles()" );
    Kokkos::fence();
    // auto sort_data = Cabana::sortByKey( keys_b );
    //  for( size_t ip=0; ip<10; ++ip)
    //  	std::cout<<keys_b(ip)<<" ";
    //  std::cout << std::endl;


    for ( size_t b = 0; b < ncell; ++b ) {
	//std::cout<<b<<" "<<npc_scan[b]<<" "<<npc_scan[b+1]<<std::endl;
        if ( npc_scan[b + 1] - npc_scan[b] <= 2 )
            continue; // do nothing
        auto sort_data = Cabana::sortByKey( keys_b, npc_scan[b], npc_scan[b + 1] );
        Cabana::permute( sort_data, particles );
        Kokkos::fence();
    }

    //std::cout<<b_max<<" "<<npc_scan[b_max]<<std::endl;
    //std::cout<<"finished sort\n";
}
