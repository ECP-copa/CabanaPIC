#include "src/input/deck.h"

// Override existing init_fields
class Custom_Field_Initializer : public Field_Initializer {
    public:
        using real_ = real_t;

        // This *has* to be virtual, as we store the object as a pointer to the
        // base class
        virtual void init(
                field_array_t& fields,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t Lx, // TODO: do we prefer xmin or Lx?
                real_t Ly,
                real_t Lz,
                real_t dx,
                real_t dy,
                real_t dz
                )
        {
            std::cout << "Using Custom field Initialization" << std::endl;

            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            real_t b0 = sqrt(20.0);
            real_t a  = 0.1;

            real_t xmin = -0.5*Lx;
            real_t ymin = -0.5*Ly;

            auto _init_fields =
                KOKKOS_LAMBDA( const int i )
                {
                    ex(i) = 0.0;
                    ey(i) = 0.0;
                    ez(i) = 0.0;
                    cbx(i) = 0.0;
                    cby(i) = 0.0;
                    cbz(i) = b0;
                    size_t ix,iy,iz;
                    RANK_TO_INDEX(i, ix,iy,iz,nx+2*ng,ny+2*ng);
                    real_t y = ymin + (iy-0.5)*dy;

                    if(y<-a) {
                        ey(i) = a;
                    }
                    else if(y>a) {
                        ey(i) =-a;
                    }
                    else {
                        ey(i) =-y;
                    }
                    //printf("%d %e %e\n",iy,y,ey(i));

                };

            Kokkos::parallel_for( fields.size(), _init_fields, "init_fields()" );
        }
};

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
                size_t ng,
                real_ dxp,
                size_t nppc,
                real_ w,
                real_ v0,
                real_ Lx, // TODO: is there a better way to pass/read global lens?
                real_ Ly,
                real_ Lz
                )
        {
            std::cout << "Using Custom Particle Initialization" << std::endl;
            std::cout << "Lx = " << Lx << " Ly " << Ly << " Lz " << Lz << std::endl;

            auto position_x = Cabana::slice<PositionX>(particles);
            auto position_y = Cabana::slice<PositionY>(particles);
            auto position_z = Cabana::slice<PositionZ>(particles);

            auto velocity_x = Cabana::slice<VelocityX>(particles);
            auto velocity_y = Cabana::slice<VelocityY>(particles);
            auto velocity_z = Cabana::slice<VelocityZ>(particles);

            auto weight = Cabana::slice<Weight>(particles);
            auto cell = Cabana::slice<Cell_Index>(particles);

            real_t hx = Lx/nx;
            real_t hy = Ly/ny;
            real_t hz = Lz/nz;
            real_t xmin = -0.5*Lx;
            real_t ymin = -0.5*Ly;

#define rand_float(min, max) (min + (max-min)*rand()/RAND_MAX)

            auto _init =
                KOKKOS_LAMBDA( const int s, const int i )
                {
                    // Initialize position.
                    size_t pi = (s)*particle_list_t::vector_length+i;
                    size_t pic = (pi)%nppc;

                    size_t ix, iy, iz;
                    real_t x, y, z;
                    x = rand_float(-0.5*Lx,0.5*Lx);
                    x= (x-xmin)/hx;
                    ix = (size_t) x;
                    x -= (real_t) ix;
                    x = x+x-1;
                    if(ix==nx) x = 1;
                    if(ix==nx) ix = nx-1;

                    y = rand_float(-0.1f, 0.1f); //a = 0.1
                    y = (y-ymin)/hy;
                    iy = (size_t) y;
                    y -= (real_t) iy;
                    y = y+y-1;
                    if(iy==ny) y = 1;
                    if(iy==ny) iy = ny-1;

                    z = 0;
                    iz = 0;

                    position_x.access(s,i) = x;
                    position_y.access(s,i) = y;
                    position_z.access(s,i) = z;

                    cell.access(s,i) = VOXEL(ix+1,iy+1,iz+1,nx,ny,nz,ng); //needs to be more general

                    weight.access(s,i) = w;

                    real_t na = 0; //0.0001*sin(2.0*3.1415926*((x+1.0+pre_ghost*2)/(2*ny)));

                    real_t gam = 1.0/sqrt(1.0-v0*v0);
                    velocity_x.access(s,i) = 0; //sign * v0 *gam*(1.0+na*sign); //0;
                    velocity_y.access(s,i) = 0;
                    velocity_z.access(s,i) = 0;

                    //std::cout << "Placing particles as "
                        //<< x << ", " << y << ", " << z << " with u=0 in cell " << cell.access(s,i) << " with w " << w << std::endl;
                };

            Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
                vec_policy( 0, particles.size() );
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );
        }
};

Input_Deck::Input_Deck()
{
    field_initer = new Custom_Field_Initializer();
    particle_initer = new Custom_Particle_Initializer();

    // User puts initialization code here
    nx = 64;
    ny = 64;
    nz = 1;

    num_steps = 20000;
    nppc = 5; // Gy has 40 and then does /8?

    v0 = 0.0;

    // Can also create temporaries
    real_ gam = 1.0 / sqrt(1.0 - v0*v0);

    const real_t default_grid_len = 1.0;

    const real_t a = 0.1;
    len_x_global = 16*a;
    len_y_global = 16*a;
    len_z_global = default_grid_len;

    Npe = n0*len_x_global*0.2*len_z_global;

    dt = 0.99*courant_length(
            len_x_global, len_y_global, len_z_global,
            nx, ny, nz
            ) / c;

    n0 = 1.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
}
