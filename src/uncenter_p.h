#ifndef uncenter_h
#define uncenter_h

void uncenter_particles(
        particle_list_t particles,
        interpolator_array_t& f0,
        real_t qdt_2mc
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

    auto _uncenter =
        //KOKKOS_LAMBDA( const int s ) {
        KOKKOS_LAMBDA( const int s, const int i ) {
            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
                // Grab particle properties
                real_t dx = position_x.access(s,i);   // Load position
                real_t dy = position_y.access(s,i);   // Load position
                real_t dz = position_z.access(s,i);   // Load position

                int ii = cell.access(s,i);

                // Grab interpolator values
                // TODO: hoist slice call?
                auto ex = f0.slice<EX>()(ii);
                auto dexdy  = f0.slice<DEXDY>()(ii);
                auto dexdz  = f0.slice<DEXDZ>()(ii);
                auto d2exdydz  = f0.slice<D2EXDYDZ>()(ii);
                auto ey  = f0.slice<EY>()(ii);
                auto deydz  = f0.slice<DEYDZ>()(ii);
                auto deydx  = f0.slice<DEYDX>()(ii);
                auto d2eydzdx  = f0.slice<D2EYDZDX>()(ii);
                auto ez  = f0.slice<EZ>()(ii);
                auto dezdx  = f0.slice<DEZDX>()(ii);
                auto dezdy  = f0.slice<DEZDY>()(ii);
                auto d2ezdxdy  = f0.slice<D2EZDXDY>()(ii);
                auto cbx  = f0.slice<CBX>()(ii);
                auto dcbxdx   = f0.slice<DCBXDX>()(ii);
                auto cby  = f0.slice<CBY>()(ii);
                auto dcbydy  = f0.slice<DCBYDY>()(ii);
                auto cbz  = f0.slice<CBZ>()(ii);
                auto dcbzdz  = f0.slice<DCBZDZ>()(ii);

                // Calculate field values
                real_t hax = qdt_2mc*(( ex + dy*dexdy ) + dz*( dexdz + dy*d2exdydz ));
                real_t hay = qdt_2mc*(( ey + dz*deydz ) + dx*( deydx + dz*d2eydzdx ));
                real_t haz = qdt_2mc*(( ez + dx*dezdx ) + dy*( dezdy + dx*d2ezdxdy ));

                cbx = cbx + dx*dcbxdx;            // Interpolate B
                cby = cby + dy*dcbydy;
                cbz = cbz + dz*dcbzdz;

                // Load momentum
                real_t ux = velocity_x.access(s,i);   // Load velocity
                real_t uy = velocity_y.access(s,i);   // Load velocity
                real_t uz = velocity_z.access(s,i);   // Load velocity

                real_t v0 = qdt_4mc/(float)sqrt(one + (ux*ux + (uy*uy + uz*uz)));

                // Borris push
                // Boris - scalars
                real_t v1 = cbx*cbx + (cby*cby + cbz*cbz);
                real_t v2 = (v0*v0)*v1;
                real_t v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
                real_t v4 = v3/(one+v1*(v3*v3));

                v4  += v4;

                v0   = ux + v3*( uy*cbz - uz*cby );      // Boris - uprime
                v1   = uy + v3*( uz*cbx - ux*cbz );
                v2   = uz + v3*( ux*cby - uy*cbx );

                ux  += v4*( v1*cbz - v2*cby );           // Boris - rotation
                uy  += v4*( v2*cbx - v0*cbz );
                uz  += v4*( v0*cby - v1*cbx );

                ux  += hax;                              // Half advance E
                uy  += hay;
                uz  += haz;

                std::cout << " hay " << hay << " ux " << ux << std::endl;

                // Store result
                velocity_x.access(s,i) = ux;
                velocity_y.access(s,i) = uy;
                velocity_z.access(s,i) = uz;

            //}
        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );
    Cabana::simd_parallel_for( vec_policy, _uncenter, "uncenter()" );
}

#endif // uncenter
