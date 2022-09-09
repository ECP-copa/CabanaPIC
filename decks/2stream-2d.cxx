#include "src/input/deck.h"
// kokkos rng
#include <Kokkos_Random.hpp>


class Custom_Field_Initializer : public Field_Initializer {

    public:
        using real_ = real_t;

        virtual void init(
                field_array_t& fields,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_ Lx, // TODO: do we prefer xmin or Lx?
                real_ Ly,
                real_ Lz,
                real_ dx,
                real_ dy,
                real_ dz
        )
        {
            std::cout << "Zero field init" << std::endl;

            // Zero fields
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto _init_fields =
                KOKKOS_LAMBDA( const int i )
                {
                    ex(i) = 0.0;
                    ey(i) = 0.0;
                    ez(i) = 0.0;
                    cbx(i) = 0.0;
                    cby(i) = 0.0;
                    cbz(i) = 0.0;
                };

            Kokkos::parallel_for( fields.size(), _init_fields, "zero_fields()" );

        }
};

// TODO: we can eventually provide a suite of default/sane initializers, such
// as ones that give the same RNG sequence over multiple procs
class Custom_Particle_Initializer : public Particle_Initializer {

    public:
        using real_ = real_t;

        virtual void init(
                particle_list_t& particles,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_ dxp,
                size_t nppc,
                real_ w,
                real_ v0,
                real_ Lx,
                real_ Ly,
                real_ Lz
                )
        {
	    size_t Np = particles.size();
	    size_t Nps = Np/2; //sqrt(Np);
	    size_t Nxp = (int) sqrt(nppc);
	    size_t Nyp = Nxp;
	    real_t dx = Lx/nx;
	    real_t dy = Ly/ny;
	    dxp = dx/Nxp;
	    real_t dyp = dy/Nyp;
            // TODO: this doesnt currently do anything with nppc/num_cells
            std::cout << "Custom particle init" << std::endl;

            auto position_x = Cabana::slice<PositionX>(particles);
            auto position_y = Cabana::slice<PositionY>(particles);
            auto position_z = Cabana::slice<PositionZ>(particles);

            auto velocity_x = Cabana::slice<VelocityX>(particles);
            auto velocity_y = Cabana::slice<VelocityY>(particles);
            auto velocity_z = Cabana::slice<VelocityZ>(particles);

            auto weight = Cabana::slice<Weight>(particles);
            auto cell = Cabana::slice<Cell_Index>(particles);

//Kokkos random number generator
	    using GeneratorPool = Kokkos::Random_XorShift64_Pool<KokkosDevice>;
	    using GeneratorType = GeneratorPool::generator_type;
	    GeneratorPool rand_pool(5374857); 

            //printf("dxp = %e \n", dxp);
            printf("part list len = %ld \n", particles.size());

            auto _init =
                KOKKOS_LAMBDA( const int i )
                {
                    // Initialize position.
                    int sign =  -1;
                    size_t pi = i;
                    //size_t pi2 = ((pi) / 2);
                    if ( pi<Nps ) {
                        sign = 1;
                    }

		    // int piy = pic/Nps;
		    // int pix = pic-piy*Nps;
		    GeneratorType rand_gen = rand_pool.get_state();
		    int pic = pi%nppc; //(int) (nppc*rand_gen.drand(1.0));
		    int iny = pic/Nxp;
		    int inx = pic - iny*Nxp;
		    real_t yy = (iny*dyp+0.5*dyp)/dy; //rand_gen.drand(1.0);
		    real_t xx = (inx*dxp+0.5*dxp)/dx; //rand_gen.drand(1.0);

                    real_t x = 2.0*xx-1; //pix*dxp+xx-1.0;
		    real_t y = 2.0*yy-1; //piy*dyp+yy-1.0;
                    int no_ghost = (pi/nppc); //pre_gohost ranges [0,nx*ny*nz).

                    int ix,iy,iz;
                    RANK_TO_INDEX(no_ghost, ix, iy, iz, nx, ny);
		    //if(i==89600) std::cout<<"no_ghost, ix,y,z="<<no_ghost<<","<<ix<<","<<iy<<","<<iz<<", nx,y,z="<<nx<<","<<ny<<","<<nz<<std::endl; 	
                    ix += ng;
                    iy += ng;
                    iz += ng;

                    position_x(i) = x;
                    position_y(i) = y;
                    position_z(i) = 0.0;

                    weight(i) = w;

                    cell(i) = VOXEL(ix,iy,iz,nx,ny,nz,ng);
                    //cell.access(s,i) = pre_ghost*(nx+2) + (nx+2)*(ny+2) + (nx+2) + 1;

                    // Initialize velocity.(each cell length is 2)
                    real_ gam = 1.0/sqrt(1.0-v0*v0);

                    real_t nax = 0.001*sin(2.0*3.1415926*((x+1.0+ix*2)/(2*nx)));
                    real_t nay = 0.001*sin(2.0*3.1415926*((y+1.0+iy*2)/(2*ny)));

                    //velocity_x.access(s,i) = sign * v0*gam; // *(1.0-na*sign); //0;
                    velocity_x(i) = sign *v0; //*gam*(1.0+nax*sign);
                    velocity_y(i) = -sign *v0; //-sign *v0*gam*(1.0+nay*sign);
                    velocity_z(i) = 0; //na*sign;  //sign * v0 *gam*(1.0+na*sign);
                    //velocity_z.access(s,i) = 1e-7*sign;

                    //printf("%d %d %d pre-g %d putting particle at y=%e with ux = %e pi = %d \n", pic, s, i, pre_ghost, position_y.access(s,i), velocity_x.access(s,i), cell.access(s,i) );
		    rand_pool.free_state(rand_gen);
                };

            Kokkos::RangePolicy<ExecutionSpace>
                vec_policy( 0, particles.size() );
            Kokkos::parallel_for( vec_policy, _init, "init()" );
        }
};

/*class _Input_Deck {
    public:
        // Having this separate lets us initialize us double if required
        using real_ = real_t;

        // I would prefer that this wasn't a pointer, but it seems to be
        // necessary. We need it to be able to do a vtable lookup for the init
        // function call, which means we need a ref or a pointer. The ref has
        // to be initialized which means the initialization of Particle_Initializer
        // would leak to the init site of _Input_Deck. A normal (non ref, non
        // pointer) variable would avoid the vtable lookup and always call the
        // "default". Perhaps there is a better way?
        // The original default pointer is pretty likely to leak in custom
        // decks... at least it does't contain a lot of state
        Particle_Initializer* particle_initer;

        Field_Initializer* field_initer;

        // Give the user a chance to hook into the end of the run, to do final
        // things like correctness checks and timing dumping
        Run_Finalizer* run_finalizer;

        _Input_Deck() :
            particle_initer(new Particle_Initializer),
            field_initer(new Field_Initializer),
            run_finalizer(new Run_Finalizer)
        {
            // empty
        }

        static real_ courant_length( real_ lx, real_ ly, real_ lz,
                size_t nx, size_t ny, size_t nz ) {
            real_ w0, w1 = 0;
            if( nx>1 ) w0 = nx/lx, w1 += w0*w0;
            if( ny>1 ) w0 = ny/ly, w1 += w0*w0;
            if( nz>1 ) w0 = nz/lz, w1 += w0*w0;
            return sqrt(1/w1);
        }

        // We could do this in the destructor, but this has 2 advantages:
        // 1) It's more explicit
        // 2) We have finer grained control, so we can more easily ensure it
        // happens before valuable data is freed
        void finalize()
        {
            run_finalizer->finalize();
        }

        void initialize_particles(
                particle_list_t& particles,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_ dxp,
                size_t nppc,
                real_ w,
                real_ v0
        )
        {
            particle_initer->init(particles, nx, ny, nz, ng, dxp, nppc, w, v0,
                    len_x_global, len_y_global, len_z_global);
        }

        void initialize_fields(
                field_array_t& fields,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_ Lx, // TODO: do we prefer xmin or Lx?
                real_ Ly,
                real_ Lz,
                real_ dx,
                real_ dy,
                real_ dz
        )
        {
            field_initer->init(
                    fields,
                    nx,
                    ny,
                    nz,
                    ng,
                    Lx,
                    Ly,
                    Lz,
                    dx,
                    dy,
                    dz
            );
        }

        real_ de = 1.0; // Length normalization (electron inertial length)
        real_ ec = 1.0; // Charge normalization
        real_ me = 1.0; // Mass normalization
        real_ mu = 1.0; // permeability of free space
        real_ c = 1.0; // Speed of light
        real_ eps = 1.0; // permittivity of free space

        real_ qsp = -ec;

        // Params
        real_ n0 = 1.0; // Background plasma density
        size_t num_species = 1;
        size_t nx = 16; // TODO: why is nx a size_t not an int?
        size_t ny = 1;
        size_t nz = 1;

        size_t num_ghosts = 1;
        size_t nppc = 1;
        real_ dt = 1.0;
        int num_steps = 2;

        // Assume domain starts at [0,0,0] and goes to [len,len,len]
        real_ len_x_global = 1.0;
        real_ len_y_global = 1.0;
        real_ len_z_global = 1.0;

        real_t Npe = -1;
        real_t Ne = -1; //(nppc*nx*ny*nz);

        //real_ local_x_min;
        //real_ local_y_min;
        //real_ local_z_min;
        //real_ local_x_max;
        //real_ local_y_max;
        //real_ local_z_max;
        real_ v0 = 1.0; //drift velocity

        //size_t ghost_offset; // Where the cell id needs to start for a "real" cell, basically nx
        //size_t num_real_cells;

        //Boundary BOUNDARY_TYPE = Boundary::Reflect;
        Boundary BOUNDARY_TYPE = Boundary::Periodic;

        ////////////////////////// DERIVED /////////////////
        // Don't set these, we can derive them instead
        real_ dx;
        real_ dy;
        real_ dz;

        real_ len_x;
        real_ len_y;
        real_ len_z;
        size_t num_cells; // This should *include* the ghost cells
        long num_particles = -1;

        bool perform_uncenter = false;

        ////////////////////////////////////////////////////

        void print_run_details()
        {
            std::cout << "#~~~ Run Specifications ~~~ " << std::endl;
            std::cout << "#Nx: " << nx << " Ny: " << ny << " Nz: " << nz << " Num Ghosts: " << num_ghosts << ". Cells Total: " << num_cells << std::endl;
            std::cout << "#Len X: " << len_x << " Len Y: " << len_y << " Len Z: " << len_z << " number of ghosts: "<<num_ghosts << std::endl;
            std::cout << "#Approx Particle Count: " << num_particles << " (nppc: " << nppc << ")" << std::endl;
            std::cout << "#~~~~~~~~~~~~~~~~~~~~~~~~~~ " << std::endl;
            std::cout << std::endl;
        }

        void derive_params()
        {
            len_x = len_x_global;
            len_y = len_y_global;
            len_z = len_z_global;

            dx = len_x / nx;
            dy = len_y / ny;
            dz = len_z / nz;

            num_cells = (nx+(2*num_ghosts)) * (ny+(2*num_ghosts)) * (nz+(2*num_ghosts));
            //num_real_cells = nx * ny * ny;

            // TODO: should we just warn the user for not setting this instead?
            if (num_particles < 0)
            {
                num_particles = nx * ny * nz * nppc; //can user define this?
                if (Ne < 0) {
                    Ne = num_particles;
                }
            }
            if (Npe < 0)
            {
                Npe = n0*len_x_global*len_y_global*len_z_global;
            }
        }

        // Function to intitialize the particles.

        void initialize_particles( particle_list_t particles,size_t nx,size_t ny,size_t nz, real_t dxp, size_t nppc, real_t w)
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

                    real_t x = pic*dxp+0.5*dxp-1.0;
                    position_x.access(s,i) = x;
                    position_y.access(s,i) = 0.;
                    position_z.access(s,i) = 0.;


                    weight.access(s,i) = w;

                    // gives me a num in the range 0..num_real_cells
                    //int pre_ghost = (s % params.num_real_cells);
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
};*/

Input_Deck::Input_Deck() {
	 // User puts initialization code here
    field_initer = new Custom_Field_Initializer();
    particle_initer = new Custom_Particle_Initializer();
	 nx = 32;
	 ny = 32;
	 nz = 1;

	 num_steps = 3000;
	 nppc = 100;

	 //v0 = 0.2;
	 v0 = 0.0866025403784439;

	 // Can also create temporaries
	 real_ gam = 1.0 / sqrt(1.0 - v0*v0);

	 const real_ default_grid_len = 1.0;

	 len_x_global = 0.628318530717959*(gam*sqrt(gam));
	 len_y_global = 0.628318530717959*(gam*sqrt(gam));
	 len_z_global = default_grid_len;

	 dt = 0.99*courant_length(
				len_x_global, len_y_global, len_z_global,
				nx, ny, nz
				) / c;
	 //dt*=10;
	 n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
}

