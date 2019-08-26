#ifndef pic_init_h
#define pic_init_h


class Initializer {
    public:
  // Compute the Courant length on a regular mesh
  static real_t courant_length( real_t lx, real_t ly, real_t lz,
				size_t nx, size_t ny, size_t nz ) {
    real_t w0, w1 = 0;
    if( nx>1 ) w0 = nx/lx, w1 += w0*w0;
    if( ny>1 ) w0 = ny/ly, w1 += w0*w0;
    if( nz>1 ) w0 = nz/lz, w1 += w0*w0;
    return sqrt(1/w1);
  }
  
  static void initialize_params(size_t _nc = 16, size_t _nppc = 16)
        {

            //logger << "Importing Default Input Deck" << std::endl;
            const real_t default_grid_len = 1.0;
	    //1D
            Parameters::instance().NX_global = _nc;
            Parameters::instance().NY_global = 1; //_nc;
            Parameters::instance().NZ_global = 1; //_nc;

            Parameters::instance().nx = Parameters::instance().NX_global;
            Parameters::instance().ny = Parameters::instance().NY_global;
            Parameters::instance().nz = Parameters::instance().NZ_global;

            Parameters::instance().num_ghosts = 1;

            Parameters::instance().num_cells =
                (Parameters::instance().nx + Parameters::instance().num_ghosts*2) *
                (Parameters::instance().ny + Parameters::instance().num_ghosts*2) *
                (Parameters::instance().nz + Parameters::instance().num_ghosts*2);

            Parameters::instance().num_real_cells =
                Parameters::instance().nx * Parameters::instance().ny * Parameters::instance().nz;

            Parameters::instance().NPPC = _nppc;

            Parameters::instance().num_particles =  Parameters::instance().NPPC  *  Parameters::instance().num_real_cells;


            Parameters::instance().num_steps = 2500;

            Parameters::instance().v0 = 0.0866025403784439;
            real_t gam = 1.0/sqrt(1.0-Parameters::instance().v0*Parameters::instance().v0);
	      
            Parameters::instance().len_x_global = 0.628318530717959*(gam*sqrt(gam)); //default_grid_len;
            Parameters::instance().len_y_global = default_grid_len;
            Parameters::instance().len_z_global = default_grid_len;

            Parameters::instance().len_x = Parameters::instance().len_x_global; //default_grid_len;
            Parameters::instance().len_y = default_grid_len;
            Parameters::instance().len_z = default_grid_len;

            Parameters::instance().dx = Parameters::instance().len_x / Parameters::instance().nx;
            Parameters::instance().dy = Parameters::instance().len_y / Parameters::instance().ny;
            Parameters::instance().dz = Parameters::instance().len_z / Parameters::instance().nz;

            Parameters::instance().dt = 0.99*courant_length(Parameters::instance().len_x,Parameters::instance().len_y,Parameters::instance().len_z,Parameters::instance().nx,Parameters::instance().ny,Parameters::instance().nz)/Parameters::instance().c;
	    Parameters::instance().n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
            Parameters::instance().print_run_details();
	    
        }

        static real_t rand_float(real_t min = 0, real_t max = 1)
        {
            return min + static_cast <real_t> (rand()) /( static_cast <real_t> (RAND_MAX/(max-min)));
        }

        // Function to intitialize the particles.
        static void initialize_particles( particle_list_t particles,size_t nx,size_t ny,size_t nz, real_t dxp, size_t nppc, real_t w)
        {
            // TODO: this doesnt currently do anything with nppc/num_cells

            auto position_x = particles.slice<PositionX>();
            auto position_y = particles.slice<PositionY>();
            auto position_z = particles.slice<PositionZ>();

            auto velocity_x = particles.slice<VelocityX>();
            auto velocity_y = particles.slice<VelocityY>();
            auto velocity_z = particles.slice<VelocityZ>();

            auto weight = particles.slice<Weight>();
            auto cell = particles.slice<Cell_Index>();

            real_t v0 = Parameters::instance().v0;

            // TODO: sensible way to do rand in parallel?
            //srand (static_cast <unsigned> (time(0)));

            auto _init =
                KOKKOS_LAMBDA( const int s, const int i )
                {
                    // Initialize position.
                    int sign =  -1;
                    size_t pi2 = (s)*particle_list_t::vector_length+i;
                    size_t pi = ((pi2) / 2);
                    if (pi2%2 == 0) {
                        sign = 1;
                    }
                    size_t pic = (2*pi)%nppc;

                    real_t x = pic*dxp+0.5*dxp-1.0; //rand_float(-1.0f, 1.0f);
                    position_x.access(s,i) = x;
                    position_y.access(s,i) = 0.; //rand_float(-1.0f, 1.0f);
                    position_z.access(s,i) = 0.; //rand_float(-1.0f, 1.0f);


                    weight.access(s,i) = w;

                    // gives me a num in the range 0..num_real_cells
                    //int pre_ghost = (s % Parameters::instance().num_real_cells);
                    //   size_t ix, iy, iz;

                    size_t pre_ghost = (2*pi/nppc);

                    cell.access(s,i) = pre_ghost + (nx+2)*(ny+2) + (nx+2) + 1; //13; //allow_for_ghosts(pre_ghost);

                    // Initialize velocity.(each cell length is 2)
                    real_t na = 0.0001*sin(2.0*3.1415926*((x+1.0+pre_ghost*2)/(2*nx)));
                    //

                    real_t gam = 1.0/sqrt(1.0-v0*v0);
                    velocity_x.access(s,i) = sign * v0 *gam*(1.0+na); //0.1;
                    velocity_y.access(s,i) = 0;
                    velocity_z.access(s,i) = 0;
                };

            Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
                vec_policy( 0, particles.size() );
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

            auto _init_interpolator =
                KOKKOS_LAMBDA( const int i )
                {
                    // Throw in some place holder values
                    ex(i) = 0.1;
                    dexdy(i) = 0.0;
                    dexdz(i) = 0.0;
                    d2exdydz(i) = 0.0;
                    ey(i) = 0.0;
                    deydz(i) = 0.0;
                    deydx(i) = 0.0;
                    d2eydzdx(i) = 0.0;
                    ez(i) = 0.0;
                    dezdx(i) = 0.0;
                    dezdy(i) = 0.0;
                    d2ezdxdy(i) = 0.0;
                    cbx(i) = 0.0;
                    dcbxdx(i) = 0.0;
                    cby(i) = 0.0;
                    dcbydy(i) = 0.0;
                    cbz(i) = 0.0;
                    dcbzdz(i) = 0.0;
                };

            Kokkos::parallel_for( f0.size(), _init_interpolator, "init_interpolator()" );

        }
};

#endif // pic_init_H
