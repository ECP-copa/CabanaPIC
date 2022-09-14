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
	    real_t xmin = 0;
	    real_t ymin = 0;
	    size_t Np = particles.size();
	    size_t Nps = Np/2; //sqrt(Np);
	    size_t nppcs = nppc/2;
	    size_t Nxp = (int) sqrt(nppcs);
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
		    //                    size_t pi2 = ((pi) / 2);
                    /*if ( pi>=Nps ) {
                        sign = 1;
			pi-=Nps;
                    }*/

		    // int piy = pic/Nps;
		    // int pix = pic-piy*Nps;
		    GeneratorType rand_gen = rand_pool.get_state();
		    real_t x               = xmin + rand_gen.drand( Lx );
		    x         = ( x - xmin ) / dx; // x is rigorously on [0,nx]
		    size_t ix = (int) x;           // ix is rigorously on [0,nx]
		    x -= (real_t) ix;              // x is rigorously on [0,1)
		    x = ( x + x ) - 1;             // x is rigorously on [-1,1)
		    if ( ix == nx ) {
			x  = 1;      // On far wall ... conditional move
			ix = nx - 1; // On far wall ... conditional move
		    }
		    ix += ng;
		    
		    real_t y               = xmin + rand_gen.drand( Ly );
		    y         = ( y - ymin ) / dy; // y is rigorously on [0,ny]
		    size_t iy = (int) y;           // iy is rigorously on [0,ny]
		    y -= (real_t) iy;              // y is rigorously on [0,1)
		    y = ( y + y ) - 1;             // y is rigorously on [-1,1)
		    if ( iy == ny ) {
			y  = 1;      // On far wall ... conditional move
			iy = ny - 1; // On far wall ... conditional move
		    }
		    iy += ng;
		    
		    size_t iz = 1;
		    real_t z = 0;
		    
		    cell( i ) = VOXEL( ix, iy, iz, nx, ny, nz, ng ); // needs to be more general
		    
                    position_x(i) = x;
                    position_y(i) = y;
                    position_z(i) = z;

                    weight(i) = w;

                    cell(i) = VOXEL(ix,iy,iz,nx,ny,nz,ng);
                    //cell.access(s,i) = pre_ghost*(nx+2) + (nx+2)*(ny+2) + (nx+2) + 1;

                    // Initialize velocity.(each cell length is 2)
                    real_ gam = 1.0/sqrt(1.0-v0*v0);

                    real_t nax = 0.5*sin(2.0*3.1415926*((x+1.0+ix*2)/(2*nx)));
                    real_t nay = 0.4*sin(2.0*3.1415926*((y+1.0+iy*2)/(2*ny)));

                    //velocity_x.access(s,i) = sign * v0*gam; // *(1.0-na*sign); //0;
                    velocity_x(i) = sign *rand_gen.normal() *v0 *gam*(1.0+nax*sign);
                    velocity_y(i) = sign *rand_gen.normal() *v0 *gam*(1.0+nay*sign);
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


Input_Deck::Input_Deck() {
	 // User puts initialization code here
    field_initer = new Custom_Field_Initializer();
    particle_initer = new Custom_Particle_Initializer();
	 nx = 64;
	 ny = 64;
	 nz = 1;

	 num_steps = 200;
	 nppc = 800;

	 //v0 = 0.2;
	 v0 = 0.0866025403784439;

	 // Can also create temporaries
	 real_ gam = 1.0 / sqrt(1.0 - v0*v0);

	 const real_ default_grid_len = 1.0;

	 len_x_global = 2.*0.628318530717959*(gam*sqrt(gam));
	 len_y_global = 2.*0.628318530717959*(gam*sqrt(gam));
	 len_z_global = default_grid_len;

	 dt = 0.99*courant_length(
				len_x_global, len_y_global, len_z_global,
				nx, ny, nz
				) / c;
	 //dt*=10;
	 n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
}

