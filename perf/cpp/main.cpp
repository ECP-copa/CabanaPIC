#include <chrono>
#include <Cabana_Core.hpp> // Using this to get Kokkos lambda

using real_t = float;

struct particle_t {
    real_t dx;
    real_t dy;
    real_t dz;
    real_t ux;
    real_t uy;
    real_t uz;
    real_t w;
    int i;
};

typedef struct interpolator
{
  real_t ex, dexdy, dexdz, d2exdydz;
  real_t ey, deydz, deydx, d2eydzdx;
  real_t ez, dezdx, dezdy, d2ezdxdy;
  real_t cbx, dcbxdx;
  real_t cby, dcbydy;
  real_t cbz, dcbzdz;
  // TODO: manuall pad?
} interpolator_t;

// TODO: aligned alloc?

int main(int argc, char *argv[])
{
    auto start_time = std::chrono::system_clock::now();

    // Init Data
    // TODO: read from command line?
    int nx = 64;
    int ny = 64;
    int nz = 64;
    int nppc = 100;

    int num_cells = nx * ny * nz;
    int np = nx*ny*nz * nppc;

    // Consts
    real_t c = 1.0; // Speed of light
    real_t ec = 1.0; // Charge normalization
    real_t qsp = -ec;
    real_t me = 1.0;
    real_t dt = 1.954867e-02;
    real_t qdt_2mc = qsp*dt/(2*me*c);

    // Create data
    particle_t* particles = new particle_t[np];
    interpolator_t* interpolator = new interpolator_t[num_cells];

    // Initialize interpolators
    for (int i = 0; i < num_cells; i++)
    {
        interpolator[i].ex = 0.0;
        interpolator[i].dexdy = 0.0;
        interpolator[i].dexdz = 0.0;
        interpolator[i].d2exdydz = 0.0;
        interpolator[i].ey = 0.0;
        interpolator[i].deydz = 0.0;
        interpolator[i].deydx = 0.0;
        interpolator[i].d2eydzdx = 0.0;
        interpolator[i].ez = 0.0;
        interpolator[i].dezdx = 0.0;
        interpolator[i].dezdy = 0.0;
        interpolator[i].d2ezdxdy = 0.0;
        interpolator[i].cbx = 0.0;
        interpolator[i].dcbxdx = 0.0;
        interpolator[i].cby = 0.0;
        interpolator[i].dcbydy = 0.0;
        interpolator[i].cbz = 0.0;
        interpolator[i].dcbzdz = 0.0;
    }

    // Place particles
    for (int x = 0; x < nx; x++)
    {
        for (int y = 0; y < ny; y++)
        {
            for (int z = 0; z < nz; z++)
            {
                for (int j = 0; j < nppc; j++)
                {
                    int index = nx * ny * nz + j;

                    // Distribute between -0.5 to 0.5
                    particles[index].dx = (1.0 / nppc) - 0.5;
                    particles[index].dy = 0.0;
                    particles[index].dz = 0.0;

                    particles[index].ux = 0.01;
                    particles[index].uy = 0.01;
                    particles[index].uz = 0.01;

                    particles[index].w = 1.0;
                    particles[index].i = x*y*z;
                }
            }
        }
    }

    const float one = 1.0f;
    const float one_third = 1.0f/3.0f;
    const float two_fifteenths = 2.0f/15.0f;

    // Define Kernel
    auto _push =
        KOKKOS_LAMBDA( const int i )
        {
            real_t dx = particles[i].dx;   // Load position
            real_t dy = particles[i].dy;   // Load position
            real_t dz = particles[i].dz;   // Load position

            int ii = particles[i].i;

            auto& ex = interpolator[ii].ex;
            auto& ey = interpolator[ii].ey;
            auto& ez = interpolator[ii].ez;

            auto& dexdy = interpolator[ii].dexdy;
            auto& dexdz = interpolator[ii].dexdz;
            auto& d2exdydz = interpolator[ii].d2exdydz;

            auto& deydz = interpolator[ii].deydz;
            auto& deydx = interpolator[ii].deydx;
            auto& d2eydzdx = interpolator[ii].d2eydzdx;

            auto& dezdx = interpolator[ii].dezdx;
            auto& dezdy = interpolator[ii].dezdy;
            auto& d2ezdxdy = interpolator[ii].d2ezdxdy;

            auto& dcbxdx = interpolator[ii].dcbxdx;
            auto& dcbydy = interpolator[ii].dcbydy;
            auto& dcbzdz = interpolator[ii].dcbzdz;

            auto& _cbx = interpolator[ii].cbx;
            auto& _cby = interpolator[ii].cby;
            auto& _cbz = interpolator[ii].cbz;

            real_t hax  = qdt_2mc*(    ( ex    + dy*dexdy    ) +
                    dz*( dexdz + dy*d2exdydz ) );
            real_t hay  = qdt_2mc*(    ( ey    + dz*deydz    ) +
                    dx*( deydx + dz*d2eydzdx ) );
            real_t haz  = qdt_2mc*(    ( ez    + dx*dezdx    ) +
                    dy*( dezdy + dx*d2ezdxdy ) );

            real_t cbx  = _cbx + dx*dcbxdx;             // Interpolate B
            real_t cby  = _cby + dy*dcbydy;
            real_t cbz  = _cbz + dz*dcbzdz;

            real_t ux = particles[i].ux;   // Load velocity
            real_t uy = particles[i].uy;   // Load velocity
            real_t uz = particles[i].uz;   // Load velocity

            ux  += hax;                               // Half advance E
            uy  += hay;
            uz  += haz;

            real_t v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
            /**/                                      // Boris - scalars
            real_t v1   = cbx*cbx + (cby*cby + cbz*cbz);
            real_t v2   = (v0*v0)*v1;
            real_t v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
            real_t v4   = v3/(one+v1*(v3*v3));
            v4  += v4;
            v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
            v1   = uy + v3*( uz*cbx - ux*cbz );
            v2   = uz + v3*( ux*cby - uy*cbx );
            ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
            uy  += v4*( v2*cbx - v0*cbz );
            uz  += v4*( v0*cby - v1*cbx );
            ux  += hax;                               // Half advance E
            uy  += hay;
            uz  += haz;

            particles[i].ux = ux;
            particles[i].uy = uy;
            particles[i].uz = uz;
        };

    auto start_kernel = std::chrono::system_clock::now();
    // Run Kernel
    for (int i = 0; i < np; i++)
    {
        _push(i);
    }
    auto end_kernel = std::chrono::system_clock::now();
    auto kernel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_kernel - start_kernel).count() / 1000.0;
    std::cout << "> Kernel runtime " << kernel_time << " seconds " << std::endl;

    auto end_time = std::chrono::system_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    std::cout << "Total runtime " << total_time << " seconds " << std::endl;
}
