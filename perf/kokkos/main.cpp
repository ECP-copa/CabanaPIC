#include <Cabana_Core.hpp> // Using this to get Kokkos lambda

using real_t = float;

/*
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
*/

const int PARTICLE_VAR_COUNT = 7;
using k_particles_t = Kokkos::View<float *[PARTICLE_VAR_COUNT]>;
using k_particles_i_t = Kokkos::View<int*>;

namespace particle_var {
  enum p_v {
    dx = 0,
    dy,
    dz,
    //pi = 3,
    ux,
    uy,
    uz,
    w,
  };
};
/*
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
*/

const int INTERPOLATOR_VAR_COUNT = 18;
using k_interpolator_t = Kokkos::View<float *[INTERPOLATOR_VAR_COUNT]>;

namespace interpolator_var {
  enum i_r {
    ex       = 0,
    dexdy    = 1,
    dexdz    = 2,
    d2exdydz = 3,
    ey       = 4,
    deydz    = 5,
    deydx    = 6,
    d2eydzdx = 7,
    ez       = 8,
    dezdx    = 9,
    dezdy    = 10,
    d2ezdxdy = 11,
    cbx      = 12,
    dcbxdx   = 13,
    cby      = 14,
    dcbydy   = 15,
    cbz      = 16,
    dcbzdz   = 17
  };
};

// TODO: aligned alloc?

int main(int argc, char *argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);
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
    k_particles_t particles("particles", np);
    k_particles_i_t particles_i("particles_i", np);
    k_interpolator_t interpolator("interpolator", num_cells);

    // Initialize interpolators
    for (int i = 0; i < num_cells; i++)
    {
        interpolator(i, interpolator_var::ex ) = 0.0;
        interpolator(i, interpolator_var::dexdy ) = 0.0;
        interpolator(i, interpolator_var::dexdz ) = 0.0;
        interpolator(i, interpolator_var::d2exdydz ) = 0.0;
        interpolator(i, interpolator_var::ey ) = 0.0;
        interpolator(i, interpolator_var::deydz ) = 0.0;
        interpolator(i, interpolator_var::deydx ) = 0.0;
        interpolator(i, interpolator_var::d2eydzdx ) = 0.0;
        interpolator(i, interpolator_var::ez ) = 0.0;
        interpolator(i, interpolator_var::dezdx ) = 0.0;
        interpolator(i, interpolator_var::dezdy ) = 0.0;
        interpolator(i, interpolator_var::d2ezdxdy ) = 0.0;
        interpolator(i, interpolator_var::cbx ) = 0.0;
        interpolator(i, interpolator_var::dcbxdx ) = 0.0;
        interpolator(i, interpolator_var::cby ) = 0.0;
        interpolator(i, interpolator_var::dcbydy ) = 0.0;
        interpolator(i, interpolator_var::cbz ) = 0.0;
        interpolator(i, interpolator_var::dcbzdz ) = 0.0;
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
                    particles(index, particle_var::dx) = (1.0 / nppc) - 0.5;
                    particles(index, particle_var::dy) = 0.0;
                    particles(index, particle_var::dz) = 0.0;

                    particles(index, particle_var::ux) = 0.01;
                    particles(index, particle_var::uy) = 0.01;
                    particles(index, particle_var::uz) = 0.01;

                    particles(index, particle_var::w) = 1.0;
                    particles_i(index) = x*y*z;
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
            real_t dx = particles(i, particle_var::dx);   // Load position
            real_t dy = particles(i, particle_var::dy);   // Load position
            real_t dz = particles(i, particle_var::dz);   // Load position

            real_t ux = particles(i, particle_var::ux);   // Load velocity
            real_t uy = particles(i, particle_var::uy);   // Load velocity
            real_t uz = particles(i, particle_var::uz);   // Load velocity

            int ii = particles_i(i);

            auto& ex = interpolator(ii, interpolator_var::ex);
            auto& ey = interpolator(ii, interpolator_var::ey);
            auto& ez = interpolator(ii, interpolator_var::ez);

            auto& dexdy = interpolator(ii, interpolator_var::dexdy);
            auto& dexdz = interpolator(ii, interpolator_var::dexdz);
            auto& d2exdydz = interpolator(ii, interpolator_var::d2exdydz);

            auto& deydz = interpolator(ii, interpolator_var::deydz);
            auto& deydx = interpolator(ii, interpolator_var::deydx);
            auto& d2eydzdx = interpolator(ii, interpolator_var::d2eydzdx);

            auto& dezdx = interpolator(ii, interpolator_var::dezdx);
            auto& dezdy = interpolator(ii, interpolator_var::dezdy);
            auto& d2ezdxdy = interpolator(ii, interpolator_var::d2ezdxdy);

            auto& dcbxdx = interpolator(ii, interpolator_var::dcbxdx);
            auto& dcbydy = interpolator(ii, interpolator_var::dcbydy);
            auto& dcbzdz = interpolator(ii, interpolator_var::dcbzdz);

            auto& _cbx = interpolator(ii, interpolator_var::cbx);
            auto& _cby = interpolator(ii, interpolator_var::cby);
            auto& _cbz = interpolator(ii, interpolator_var::cbz);

            real_t hax  = qdt_2mc*(    ( ex    + dy*dexdy    ) +
                    dz*( dexdz + dy*d2exdydz ) );
            real_t hay  = qdt_2mc*(    ( ey    + dz*deydz    ) +
                    dx*( deydx + dz*d2eydzdx ) );
            real_t haz  = qdt_2mc*(    ( ez    + dx*dezdx    ) +
                    dy*( dezdy + dx*d2ezdxdy ) );

            real_t cbx  = _cbx + dx*dcbxdx;             // Interpolate B
            real_t cby  = _cby + dy*dcbydy;
            real_t cbz  = _cbz + dz*dcbzdz;

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

            particles(i, particle_var::ux) = ux;
            particles(i, particle_var::uy) = uy;
            particles(i, particle_var::uz) = uz;
        };

    // Run Kernel
    Kokkos::parallel_for(
        "advance_p",
        Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
        _push
    );
}
