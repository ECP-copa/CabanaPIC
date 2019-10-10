#include "src/input/deck.h"
// For a list of available global variables, see `src/input/deck.h`, common ones include:
/*
        real_ de = 1.0; // Length normalization (electron inertial length)
        real_ ec = 1.0; // Charge normalization
        real_ me = 1.0; // Mass normalization
        real_ mu = 1.0; // permeability of free space
        real_ c = 1.0; // Speed of light
        real_ eps = 1.0; // permittivity of free space
        real_ n0 = 1.0; // Background plasma density
        size_t nx = 16;
        size_t ny = 1;
        size_t nz = 1;
        size_t nppc = 1;
        double dt = 1.0;
        int num_steps = 2;
        real_ len_x_global = 1.0;
        real_ len_y_global = 1.0;
        real_ len_z_global = 1.0;
        real_ v0 = 1.0; //drift velocity
        size_t num_ghosts = 1;
        (len_x and dx will automatically be set)
*/
// I would rather decalare this as a class, not just as a constructor, but that
// would have to be in a header (which would stop the compile detecting
// changes...). This is fine for now.

// Override existing init_particles
class Custom_Particle_Initializer : public Particle_Initializer {
    public:
        using real_ = real_t;

        // This *has* to be virtual, as we store the object as a pointer to the
        // base class
        virtual void init(
                particle_list_t& particles,
                size_t nx,
                size_t ny,
                size_t nz,
                real_ dxp,
                size_t nppc,
                real_ w,
                real_ v0
                )
        {
            std::cout << "Using Custom Particle Initialization" << std::endl;

            auto position_x = Cabana::slice<PositionX>(particles);
            auto position_y = Cabana::slice<PositionY>(particles);
            auto position_z = Cabana::slice<PositionZ>(particles);

            auto velocity_x = Cabana::slice<VelocityX>(particles);
            auto velocity_y = Cabana::slice<VelocityY>(particles);
            auto velocity_z = Cabana::slice<VelocityZ>(particles);

            auto weight = Cabana::slice<Weight>(particles);
            auto cell = Cabana::slice<Cell_Index>(particles);

            // TODO: sensible way to do rand in parallel?
            //srand (static_cast <unsigned> (time(0)));

            auto _init =
                KOKKOS_LAMBDA( const int s, const int i )
                {
                    // Initialize position.
                    int sign =  -1;
                    size_t pi2 = (s)*particle_list_t::vector_length+i;
                    size_t pi = ((pi2) / 2);
                    if (pi2%2 == 1) {
                        sign = 1;
                    }
                    size_t pic = (2*pi)%nppc;

                    real_t x = pic*dxp+0.5*dxp-1.0; //rand_float(-1.0f, 1.0f); //
                    size_t pre_ghost = (2*pi/nppc);

                    position_x.access(s,i) = 0;
                    position_y.access(s,i) = x; //rand_float(-1.0f, 1.0f);
                    position_z.access(s,i) = 0; //rand_float(-1.0f, 1.0f);

                    weight.access(s,i) = w;

                    cell.access(s,i) = pre_ghost*(nx+2) + (nx+2)*(ny+2) + (nx+2) + 1;

                    real_t gam = 1.0/sqrt(1.0-v0*v0);
                    velocity_x.access(s,i) = sign * v0*gam; // *(1.0-na*sign); //0;
                    velocity_y.access(s,i) = 0;
                    velocity_z.access(s,i) = 0; //na*sign;  //sign * v0 *gam*(1.0+na*sign);
                    velocity_z.access(s,i) = 1e-8*sign;
                };

            Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
                vec_policy( 0, particles.size() );
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );
        }
};

Input_Deck::Input_Deck()
{
    // User puts initialization code here

    std::cout << "Input_Deck constructor" << std::endl;
    // Tell the deck to use the custom initer in place of the default
    particle_initer = new Custom_Particle_Initializer();

    nx = 1;
    ny = 32;
    nz = 1;

    num_steps = 30;
    nppc = 100;

    v0 = 0.2;

    // Can also create temporaries
    real_ gam = 1.0 / sqrt(1.0 - v0*v0);

    const real_t default_grid_len = 1.0;

    len_x_global = default_grid_len;
    len_y_global = 3.14159265358979*0.5; // TODO: use proper PI?
    len_z_global = default_grid_len;

    dt = 0.99*courant_length(
            len_x_global, len_y_global, len_z_global,
            nx, ny, nz
            ) / c;

    n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
}
