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


            real_t x0 = 0;
            real_t hx = Lx/nx;
            real_t xp1=0.6; //-0.5*hx;
            real_t xp2=0.8; //+0.5*hx;

            real_t phi[nx+2], Ex[nx+2];

            //real_t wi = Lx/(nx*2.0); //particle weight (how to have it here?)
            real_t wi = 1./2.;

            // 	    for (int i=0; i<nx+2; i++){
            // //       double xp = x0 + dx*i;
            // 	      double xc = x0 + dx*(i-0.5);
            // 	      if(xc>xp1&&xc<xp2)
            // 	 //phi[i] = (xp1+(1.0-xp1-xp2)*xc +xc*xc-xc)*0.5;
            // 		phi[i] = (xp1+(1.0-xp1-xp2)*xc)*wi;
            // 	      else if(xc<xp1)
            // 		phi[i] = (xc*(2.0-xp1-xp2))*wi;
            // 	      else if(xc>xp2)
            // 		phi[i] = (xp1+xp2-(xp1+xp2)*xc)*wi;

            // //       printf("%d %e %e ", i, xc,phi[i]);
            // 	    }
            // 	    for (int i=1; i<nx+1; i++){
            // 	      double xc = x0 + dx*(i-0.5);
            // 	      Ex[i] = (phi[i-1] - phi[i+1])/(2.0*dx) - (xc-0.5)*wi;
            // 	    }
            // 	    Ex[0] = Ex[nx];
            // 	    Ex[nx+1] = Ex[1];

            for (size_t i=0; i<nx+2; i++){
                real_t xn = x0 + dx*(i-1);
                if(xn>=xp1&&xn<=xp2)
                    phi[i] = (1.0-xn)*xp1*wi + xn*(1.0-xp2)*wi + (xn*xn-xn)*wi;
                else if(xn<xp1)
                    phi[i] = (1.0-xp1)*xn*wi + xn*(1.0-xp2)*wi + (xn*xn-xn)*wi;
                else if(xn>xp2)
                    phi[i] = (1.0-xn)*xp1*wi + xp2*(1.0-xn)*wi + (xn*xn-xn)*wi;
            }
            for (size_t i=1; i<nx+1; i++){
                Ex[i] = (phi[i] - phi[i+1])/(dx);
            }
            Ex[0] = Ex[nx];
            Ex[nx+1] = Ex[1];
            // for (int i=0; i<nx+2; i++){
            //   double xc = x0 + dx*(i-0.5);
            //   double xn = x0 + dx*(i-1);
            //   printf("%e %e %e %e %e\n", xn,xc,phi[i],Ex[i], (Ex[i]-Ex[i-1])/dx);
            // }
            // exit(1);


            for(size_t i=0; i<fields.size(); i++){
                ey(i) = 0.0;
                ez(i) = 0.0;
                cbx(i) = 0.0;
                cby(i) = 0.0;
                cbz(i) = 0.0;
                size_t ix,iy,iz;
                RANK_TO_INDEX(i, ix,iy,iz,nx+2*ng,ny+2*ng);
                ex(i) = Ex[ix];
                //		    printf("%d %e\n",ix, ex(i));
            }
            // auto _init_fields =
            //     KOKKOS_LAMBDA( const int i )
            //     {
            //         ex(i) = 0.0;
            //         ey(i) = 0.0;
            //         ez(i) = 0.0;
            //         cbx(i) = 0.0;
            //         cby(i) = 0.0;
            //         cbz(i) = 0.0;
            //         size_t ix,iy,iz;
            //         RANK_TO_INDEX(i, ix,iy,iz,nx+2*ng,ny+2*ng);
            // 	    ex(i) = Ex[ix]; //does not work
            //         //printf("%d %e %e\n",iy,y,ey(i));

            //     };

            // Kokkos::parallel_for( fields.size(), _init_fields, "init_fields()" );

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
            //real_t hy = Ly/ny;
            //real_t hz = Lz/nz;
            real_t xmin = 0; //-0.5*Lx;
            //real_t ymin = 0; //-0.5*Ly;

#define rand_float(min, max) (min + (max-min)*rand()/RAND_MAX)

            auto _init =
                KOKKOS_LAMBDA( const int s, const int i )
                {
                    // Initialize position.
                    size_t pi = (s)*particle_list_t::vector_length+i;
                    real_t xp1=0.6; //-0.5*hx; //those two numbers are also used in field init
                    real_t xp2=0.8; //+0.5*hx;
                    //2 particles only
                    size_t ix, iy, iz;
                    real_t x, y, z;
                    if(pi==0){
                        x = xp1;
                        x= (x-xmin)/hx;

                        ix = (size_t) x;
                        iy = 1;
                        iz = 1;
                        x = 1-0.5*hx;
                        y = 0;
                        z = 0;
                        //		      ix++;
                    }else{
                        x = xp2;
                        x= (x-xmin)/hx;
                        ix = (size_t) x;
                        iy = 1;
                        iz = 1;
                        x = -1+0.5*hx;
                        y = 0;
                        z = 0;
                        ix++;
                    }

                    position_x.access(s,i) = x;
                    position_y.access(s,i) = y;
                    position_z.access(s,i) = z;

                    cell.access(s,i) = VOXEL(ix,iy,iz,nx,ny,nz,ng); //needs to be more general

                    weight.access(s,i) = w;

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
    nx = 1000;
    ny = 1;
    nz = 1;

    num_steps = 200000;
    nppc = 1;

    v0 = 0.0;

    const real_t default_grid_len = 1.0;

    //    const real_t a = 0.1;
    len_x_global = default_grid_len; //16*a;
    len_y_global = default_grid_len; //16*a;
    len_z_global = default_grid_len;

    n0 = 1.0;
    Npe = n0*len_x_global*len_y_global*len_z_global;

    dt = 0.99*courant_length(
            len_x_global, len_y_global, len_z_global,
            nx, ny, nz
            ) / c;

    ec = 1.0;
    qsp = ec;
    me = qsp;

    Ne = 2;

    num_particles = 2;

}
