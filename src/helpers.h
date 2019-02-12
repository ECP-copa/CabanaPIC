#ifndef pic_helper_h
#define pic_helper_h

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
                std::cout << "Struct id: " << s;
                std::cout << " Struct offset: " << i;
                std::cout << " Position: "
                    << position_x.access(s,i) << " "
                    << position_y.access(s,i) << " "
                    << position_z.access(s,i) << " ";
                std::cout << std::endl;

                std::cout << "Velocity "
                    << velocity_x.access(s,i) << " "
                    << velocity_y.access(s,i) << " "
                    << velocity_z.access(s,i) << " ";
                std::cout << std::endl;
        };

    // TODO: How much sense does printing in parallel make???
    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );
    logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    Cabana::simd_parallel_for( vec_policy, _print, "_print()" );

}

#endif // pic_helper_h
