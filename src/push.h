#ifndef pic_push_h
#define pic_push_h

#include <types.h>
//#include "move_p.h"

void push(
        particle_list_t particles,
        interpolator_array_t& f0,
        real_t qdt_2mc,
        real_t cdt_dx,
        real_t cdt_dy,
        real_t cdt_dz,
        real_t qsp,
        accumulator_array_t& a0,
        grid_t* g
    )
{

    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto charge = particles.slice<Charge>();
    auto cell = particles.slice<Cell_Index>();

    const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;

    auto _push =
        KOKKOS_LAMBDA( const int s, const int i )
        {
            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
                // TODO: deal with pms
                particle_mover_t local_pm = particle_mover_t();

                real_t dx = position_x.access(s,i);   // Load position
                real_t dy = position_y.access(s,i);   // Load position
                real_t dz = position_z.access(s,i);   // Load position

                int ii = cell.access(s,i);

                // TODO: hoist slice call
                auto ex = f0.slice<EX>()(ii);
                auto dexdy = f0.slice<DEXDY>()(ii);
                auto dexdz = f0.slice<DEXDZ>()(ii);
                auto d2exdydz = f0.slice<D2EXDYDZ>()(ii);
                auto ey = f0.slice<EY>()(ii);
                auto deydz = f0.slice<DEYDZ>()(ii);
                auto deydx = f0.slice<DEYDX>()(ii);
                auto d2eydzdx = f0.slice<D2EYDZDX>()(ii);
                auto ez = f0.slice<EZ>()(ii);
                auto dezdx = f0.slice<DEZDX>()(ii);
                auto dezdy = f0.slice<DEZDY>()(ii);
                auto d2ezdxdy = f0.slice<D2EZDXDY>()(ii);
                auto cbx = f0.slice<CBX>()(ii);
                auto dcbxdx = f0.slice<DCBXDX>()(ii);
                auto cby = f0.slice<CBY>()(ii);
                auto dcbydy = f0.slice<DCBYDY>()(ii);
                auto cbz = f0.slice<CBZ>()(ii);
                auto dcbzdz = f0.slice<DCBZDZ>()(ii);
                /*
                auto ex  = f0.get<EX>(ii);
                auto dexdy  = f0.get<DEXDY>(ii);
                auto dexdz  = f0.get<DEXDZ>(ii);
                auto d2exdydz  = f0.get<D2EXDYDZ>(ii);
                auto ey  = f0.get<EY>(ii);
                auto deydz  = f0.get<DEYDZ>(ii);
                auto deydx  = f0.get<DEYDX>(ii);
                auto d2eydzdx  = f0.get<D2EYDZDX>(ii);
                auto ez  = f0.get<EZ>(ii);
                auto dezdx  = f0.get<DEZDX>(ii);
                auto dezdy  = f0.get<DEZDY>(ii);
                auto d2ezdxdy  = f0.get<D2EZDXDY>(ii);
                auto cbx  = f0.get<CBX>(ii);
                auto dcbxdx   = f0.get<DCBXDX>(ii);
                auto cby  = f0.get<CBY>(ii);
                auto dcbydy  = f0.get<DCBYDY>(ii);
                auto cbz  = f0.get<CBZ>(ii);
                auto dcbzdz  = f0.get<DCBZDZ>(ii);
                */

                /*
                real_t hax  = qdt_2mc*(    ( f.ex    + dy*f.dexdy    ) +
                        dz*( f.dexdz + dy*f.d2exdydz ) );
                real_t hay  = qdt_2mc*(    ( f.ey    + dz*f.deydz    ) +
                        dx*( f.deydx + dz*f.d2eydzdx ) );
                real_t haz  = qdt_2mc*(    ( f.ez    + dx*f.dezdx    ) +
                        dy*( f.dezdy + dx*f.d2ezdxdy ) );
                */
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

                real_t q = charge.access(s,i);   // Load velocity

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

                v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));
                /**/                                      // Get norm displacement
                ux  *= cdt_dx;
                uy  *= cdt_dy;
                uz  *= cdt_dz;
                ux  *= v0;
                uy  *= v0;
                uz  *= v0;
                v0   = dx + ux;                           // Streak midpoint (inbnds)
                v1   = dy + uy;
                v2   = dz + uz;
                v3   = v0 + ux;                           // New position
                v4   = v1 + uy;
                real_t v5   = v2 + uz;

                // Check if inbnds
                //if(  v3<=one &&  v4<=one &&  v5<=one && -v3<=one && -v4<=one && -v5<=one )
                {

                    // Common case (inbnds).  Note: accumulator values are 4 times
                    // the total physical charge that passed through the appropriate
                    // current quadrant in a time-step

                    q *= qsp;

                    // Store new position
                    position_x.access(s,i) = v3;
                    position_y.access(s,i) = v4;
                    position_z.access(s,i) = v5;

                    dx = v0;                                // Streak midpoint
                    dy = v1;
                    dz = v2;
                    v5 = q*ux*uy*uz*one_third;              // Compute correction

                    //real_t* a  = (real_t *)( a0[ii].a );              // Get accumulator

#     define ACCUMULATE_J(X,Y,Z,offset)                                 \
                    v4  = q*u##X;   /* v2 = q ux                            */        \
                    v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
                    v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
                    v1 += v4;       /* v1 = q ux (1+dy)                     */        \
                    v4  = one+d##Z; /* v4 = 1+dz                            */        \
                    v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
                    v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
                    v4  = one-d##Z; /* v4 = 1-dz                            */        \
                    v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
                    v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
                    v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
                    v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
                    v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
                    v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */        \
                    a0.slice<0>()(ii,offset+0) += v0; \
                    a0.slice<0>()(ii,offset+1) += v1; \
                    a0.slice<0>()(ii,offset+2) += v2; \
                    a0.slice<0>()(ii,offset+3) += v3;

                    ACCUMULATE_J( x,y,z, 0 );
                    ACCUMULATE_J( y,z,x, 4 );
                    ACCUMULATE_J( z,x,y, 8 );

#     undef ACCUMULATE_J

                }

                /*
                else
                {                                    // Unlikely
                    local_pm.dispx = ux;
                    local_pm.dispy = uy;
                    local_pm.dispz = uz;

                    local_pm.i = s*particle_list_t::vector_length + i; //i + itmp; //p_ - p0;

                    // TODO: renable this
                    //if( move_p( p0, local_pm, a0, g, qsp ) ) { // Unlikely
                        //if( nm<max_nm ) {
                            //pm[nm++] = local_pm[0];
                        //}
                        //else {
                            //ignore++;                 // Unlikely
                        //} // if
                    //} // if
                }
                */

            //}
        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );
    Cabana::simd_parallel_for( vec_policy, _push, "push()" );
}

#endif // pic_push_h
