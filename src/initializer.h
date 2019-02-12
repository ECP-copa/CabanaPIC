#ifndef pic_init_h
#define pic_init_h


class Initializer {
    public:
        static void initialize_params(size_t _nc = 16, size_t _nppc = 16)
        {

            logger << "Importing Default Input Deck" << std::endl;
            const real_t default_grid_len = 1.0;

            Parameters::instance().NX_global = _nc;
            Parameters::instance().NY_global = _nc;
            Parameters::instance().NZ_global = _nc;

            Parameters::instance().nx = Parameters::instance().NX_global;
            Parameters::instance().ny = Parameters::instance().NY_global;
            Parameters::instance().nz = Parameters::instance().NZ_global;

            Parameters::instance().num_ghosts = 1;

            Parameters::instance().num_cells =
                (Parameters::instance().nx + Parameters::instance().num_ghosts*2) *
                (Parameters::instance().ny + Parameters::instance().num_ghosts*2) *
                (Parameters::instance().nz + Parameters::instance().num_ghosts*2);

            Parameters::instance().NPPC = _nppc;

            Parameters::instance().num_particles =  Parameters::instance().NPPC  *  Parameters::instance().num_cells;

            Parameters::instance().dt = 0.1;

            Parameters::instance().num_steps = 5;

            Parameters::instance().len_x_global = default_grid_len;
            Parameters::instance().len_y_global = default_grid_len;
            Parameters::instance().len_z_global = default_grid_len;

            Parameters::instance().len_x = default_grid_len;
            Parameters::instance().len_y = default_grid_len;
            Parameters::instance().len_z = default_grid_len;

            Parameters::instance().dx = Parameters::instance().len_x / Parameters::instance().nx;
            Parameters::instance().dy = Parameters::instance().len_y / Parameters::instance().ny;
            Parameters::instance().dz = Parameters::instance().len_z / Parameters::instance().nz;

            Parameters::instance().print_run_details();
        }

        // Function to intitialize the particles.
        static void initialize_particles( particle_list_t particles )
        {
            auto position_x = particles.slice<PositionX>();
            auto position_y = particles.slice<PositionY>();
            auto position_z = particles.slice<PositionZ>();

            auto velocity_x = particles.slice<VelocityX>();
            auto velocity_y = particles.slice<VelocityY>();
            auto velocity_z = particles.slice<VelocityZ>();

            auto charge = particles.slice<Charge>();
            auto cell = particles.slice<Cell_Index>();

            auto _init =
                //KOKKOS_LAMBDA( const int s )
                KOKKOS_LAMBDA( const int s, const int i )
                {
                    // Much more likely to vectroize and give good performance
                    float counter = (float)s;
                    //for ( int i = 0; i < particle_list_t::vector_length; ++i )
                    //{
                    // Initialize position.
                    position_x.access(s,i) = 1.1 + counter;
                    position_y.access(s,i) = 2.2 + counter;
                    position_z.access(s,i) = 3.3 + counter;

                    // Initialize velocity.
                    velocity_x.access(s,i) = 0.1;
                    velocity_y.access(s,i) = 0.2;
                    velocity_z.access(s,i) = 0.3;

                    charge.access(s,i) = 1.0;
                    cell.access(s,i) = s;
                    counter += 0.01;
                    //}
                };
            Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
                vec_policy( 0, particles.numSoA() );
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );
        }



        static void initialize_interpolator(interpolator_array_t& f0)
        {
            auto ex = f0.slice<EX>();
            auto dexdy  = f0.slice<DEXDY>();
            auto dexdz  = f0.slice<DEXDZ>();
            auto d2exdydz  = f0.slice<D2EXDYDZ>();
            auto ey  = f0.slice<EY>();
            auto deydz  = f0.slice<DEYDZ>();
            auto deydx  = f0.slice<DEYDX>();
            auto d2eydzdx  = f0.slice<D2EYDZDX>();
            auto ez  = f0.slice<EZ>();
            auto dezdx  = f0.slice<DEZDX>();
            auto dezdy  = f0.slice<DEZDY>();
            auto d2ezdxdy  = f0.slice<D2EZDXDY>();
            auto cbx  = f0.slice<CBX>();
            auto dcbxdx   = f0.slice<DCBXDX>();
            auto cby  = f0.slice<CBY>();
            auto dcbydy  = f0.slice<DCBYDY>();
            auto cbz  = f0.slice<CBZ>();
            auto dcbzdz  = f0.slice<DCBZDZ>();

            for (size_t i = 0; i < f0.size(); i++)
            {
                // Throw in some place holder values
                ex(i) = 0.01;
                dexdy(i) = 0.02;
                dexdz(i) = 0.03;
                d2exdydz(i) = 0.04;
                ey(i) = 0.05;
                deydz(i) = 0.06;
                deydx(i) = 0.07;
                d2eydzdx(i) = 0.08;
                ez(i) = 0.09;
                dezdx(i) = 0.10;
                dezdy(i) = 0.11;
                d2ezdxdy(i) = 0.12;
                cbx(i) = 0.13;
                dcbxdx(i) = 0.14;
                cby(i) = 0.15;
                dcbydy(i) = 0.16;
                cbz(i) = 0.17;
                dcbzdz(i) = 0.18;
            }
        }
};

#endif // pic_init_H
