#include <iostream>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#ifdef USE_GPU
using MemorySpace = Kokkos::CudaSpace;
using ExecutionSpace = Kokkos::Cuda;
#else
  #ifdef USE_SERIAL_CPU
    //cpu
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
  #else // CPU Parallel
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace; //Kokkos::OpenMP;
  #endif
#endif

int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Kokkos::ScopeGuard scope_guard( argc, argv );

    {
        printf("#Running On Kokkos execution space %s\n",
                typeid (Kokkos::DefaultExecutionSpace).name ());

        // Create Cartesian grid topology
        int comm_size = -1;

        MPI_Init(&argc, &argv);
        MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

        std::cout << "Start" << std::endl;

        // Create the MPI partitions.
        Cajita::UniformDimPartitioner partitioner;
        std::array<int, 3> ranks_per_dim = partitioner.ranksPerDimension( MPI_COMM_WORLD, {} );

        std::array<double, 3> low_corner = { -1.0, -1.0, -1.0 };
        std::array<double, 3> high_corner = { 1.0, 1.0, 1.0 };

        // Create global mesh of MPI partitions.
        auto global_mesh = Cajita::createUniformGlobalMesh(
                low_corner, high_corner, ranks_per_dim );

        std::array<bool, 3> is_periodic = {true, true, true};

        // Create the global grid.
        auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                is_periodic, partitioner );

        int halo_width = 1;
        auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );
        auto local_mesh = Cajita::createLocalMesh<Kokkos::DefaultExecutionSpace>( *local_grid );

        double local_mesh_lo_x = local_mesh.lowCorner( Cajita::Own(), 0 );
        double local_mesh_lo_y = local_mesh.lowCorner( Cajita::Own(), 1 );
        double local_mesh_lo_z = local_mesh.lowCorner( Cajita::Own(), 2 );
        double local_mesh_hi_x = local_mesh.highCorner( Cajita::Own(), 0 );
        double local_mesh_hi_y = local_mesh.highCorner( Cajita::Own(), 1 );
        double local_mesh_hi_z = local_mesh.highCorner( Cajita::Own(), 2 );
        double ghost_mesh_lo_x = local_mesh.lowCorner( Cajita::Ghost(), 0 );

        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );

        std::cout << rank << " x lo " << local_mesh_lo_x << " hi " << local_mesh_hi_x << std::endl;

        using ParticleDataTypes =
            Cabana::MemberTypes<
            float,                        // (0) x-position
            float,                        // (1) y-position
            float                        // (2) z-position
        >;
        using particle_list_t =
            Cabana::AoSoA<ParticleDataTypes,MemorySpace>;

        int num_particles = 100;

        particle_list_t particles( "particles", num_particles );
        particle_list_t leavers( "particles", num_particles );

        auto position_x = Cabana::slice<0>(particles);
        auto position_y = Cabana::slice<1>(particles);
        auto position_z = Cabana::slice<2>(particles);

        auto leavers_x = Cabana::slice<0>(leavers);
        auto leavers_y = Cabana::slice<1>(leavers);
        auto leavers_z = Cabana::slice<2>(leavers);

        const float HI = local_mesh_hi_x;
        const float LO = local_mesh_lo_x;

        srand( rank*10 );

        // init them
        auto _init =
            KOKKOS_LAMBDA( const int s, const int i )
            {
                // TODO: use kokkos rng
                float x = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                position_x.access(s,i) = x;
                position_y.access(s,i) = 0.0;
                position_z.access(s,i) = 0.0;
            };
        Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
            vec_policy( 0, particles.size() );
        Cabana::simd_parallel_for( vec_policy, _init, "init()" );


        float v_move = (HI-LO) / 10.0; // try get 10%

        Kokkos::View<int> over_count("over count");
        Kokkos::View<int*> export_ranks("export ranks", num_particles);

        Kokkos::deep_copy( export_ranks, -1 ); // default to not sending

        int neighbors_recv[6]; // Neighbor for each phase
        int neighbors_send[6]; // Neighbor for each phase

        neighbors_send[0] = local_grid->neighborRank( 1, 0, 0 );
        neighbors_send[1] = local_grid->neighborRank( -1, 0, 0 );
        neighbors_send[2] = local_grid->neighborRank( 0, 1, 0 );
        neighbors_send[3] = local_grid->neighborRank( 0, -1, 0 );
        neighbors_send[4] = local_grid->neighborRank( 0, 0, 1 );
        neighbors_send[5] = local_grid->neighborRank( 0, 0, -1 );

        neighbors_recv[0] = neighbors_send[1];
        neighbors_recv[1] = neighbors_send[0];
        neighbors_recv[2] = neighbors_send[3];
        neighbors_recv[3] = neighbors_send[2];
        neighbors_recv[4] = neighbors_send[5];
        neighbors_recv[5] = neighbors_send[4];

        std::cout << rank << " Neighbors: " <<
            " 1, 0, 0 = " << neighbors_send[0] <<
            ".     -1, 0, 0 = " << neighbors_send[1] <<
            ".     0, 1, 0 = " << neighbors_send[2] <<
            ".     0, -1, 0 = " << neighbors_send[3] <<
            ".     0, 0, 1 = " << neighbors_send[4] <<
            ".     0, 0, -1 = " << neighbors_send[5] <<
            std::endl;

        // move them
        auto _move =
            KOKKOS_LAMBDA( const int s, const int i )
            {
                position_x.access(s,i) += v_move;

                if (position_x.access(s, i) > HI)
                {
                    int cur = over_count();
                    over_count()++;
                    leavers_x( cur ) = position_x.access(s, i);
                    leavers_y( cur ) = position_y.access(s, i);
                    leavers_z( cur ) = position_z.access(s, i);


                    int neighbor_rank = neighbors_send[0];
                    export_ranks( cur ) = neighbor_rank;
                }
            };

        Cabana::simd_parallel_for( vec_policy, _move, "move()" );

        std::cout << rank << " has " << over_count() << std::endl;

        for (int i = 0; i < over_count()+2; i++)
        {
            std::cout << i << " = " << leavers_x(i) << std::endl;
        }

        // TODO: over count needs to be on the host
        // Try shrink it so it only iterates over the required vals, as we know
        // we allocated too much space
        //leavers.resize( over_count() );
        //export_ranks.resize( over_count() );
        // We can only resize AoSoAs not views

        //static MPI_Comm grid_comm;
        //MPI_Cart_create( MPI_COMM_WORLD, 3, dims, wrap, false, &grid_comm );

        auto particle_distributor = Cabana::Distributor<MemorySpace>(
                MPI_COMM_WORLD, export_ranks );

        std::cout << rank << " expecting to recv " << particle_distributor.totalNumImport() << std::endl;

        Cabana::migrate( particle_distributor, leavers );

        auto _print =
            KOKKOS_LAMBDA( const int s, const int i )
            {
                float x = leavers_x.access(s, i);
                if (x > 1.0)
                {
                    printf("%d at pos %e \n", rank, x );
                }
            };

        // Backfill leavers into the main population

        Cabana::simd_parallel_for( vec_policy, _print, "print()" );

    }

    MPI_Finalize();
}

