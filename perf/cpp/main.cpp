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

int main()
{
    // Init Data
    // TODO

    // Define Kernel
    auto _push =
        KOKKOS_LAMBDA( const int i )
        {
            real_t dx = position_x.access(s,i);   // Load position
            real_t dy = position_y.access(s,i);   // Load position
            real_t dz = position_z.access(s,i);   // Load position

            real_t hax  = qdt_2mc*(    ( ex    + dy*dexdy    ) +
                    dz*( dexdz + dy*d2exdydz ) );
            real_t hay  = qdt_2mc*(    ( ey    + dz*deydz    ) +
                    dx*( deydx + dz*d2eydzdx ) );
            real_t haz  = qdt_2mc*(    ( ez    + dx*dezdx    ) +
                    dy*( dezdy + dx*d2ezdxdy ) );

            cbx  = cbx + dx*dcbxdx;             // Interpolate B
            cby  = cby + dy*dcbydy;
            cbz  = cbz + dz*dcbzdz;

            real_t ux = velocity_x.access(s,i);   // Load velocity
            real_t uy = velocity_y.access(s,i);   // Load velocity
            real_t uz = velocity_z.access(s,i);   // Load velocity

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

            velocity_x.access(s,i) = ux;
            velocity_y.access(s,i) = uy;
            velocity_z.access(s,i) = uz;
        };

    // Run Kernel
}
