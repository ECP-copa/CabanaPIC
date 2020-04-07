#ifndef uncenter_h
#define uncenter_h

void uncenter_particles(
        particle_list_t particles,
        interpolator_array_t& f0,
        real_t qdt_2mc
    )
{

    auto position_x = Cabana::slice<PositionX>(particles);
    auto position_y = Cabana::slice<PositionY>(particles);
    auto position_z = Cabana::slice<PositionZ>(particles);

    auto velocity_x = Cabana::slice<VelocityX>(particles);
    auto velocity_y = Cabana::slice<VelocityY>(particles);
    auto velocity_z = Cabana::slice<VelocityZ>(particles);

    //auto weight = Cabana::slice<Weight>(particles);
    auto cell = Cabana::slice<Cell_Index>(particles);

    const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;

    auto _uncenter =
        //KOKKOS_LAMBDA( const int s ) {
        KOKKOS_LAMBDA( const int s, const int i ) {
            // Grab particle properties
            real_t dx = position_x.access(s,i);   // Load position
            real_t dy = position_y.access(s,i);   // Load position
            real_t dz = position_z.access(s,i);   // Load position

            int ii = cell.access(s,i);

            // Grab interpolator values
            // TODO: hoist slice call?
            auto ex       = Cabana::slice<EX>(f0)(ii);
            auto dexdy    = Cabana::slice<DEXDY>(f0)(ii);
            auto dexdz    = Cabana::slice<DEXDZ>(f0)(ii);
            auto d2exdydz = Cabana::slice<D2EXDYDZ>(f0)(ii);
            auto ey       = Cabana::slice<EY>(f0)(ii);
            auto deydz    = Cabana::slice<DEYDZ>(f0)(ii);
            auto deydx    = Cabana::slice<DEYDX>(f0)(ii);
            auto d2eydzdx = Cabana::slice<D2EYDZDX>(f0)(ii);
            auto ez       = Cabana::slice<EZ>(f0)(ii);
            auto dezdx    = Cabana::slice<DEZDX>(f0)(ii);
            auto dezdy    = Cabana::slice<DEZDY>(f0)(ii);
            auto d2ezdxdy = Cabana::slice<D2EZDXDY>(f0)(ii);
            auto cbx      = Cabana::slice<CBX>(f0)(ii);
            auto dcbxdx   = Cabana::slice<DCBXDX>(f0)(ii);
            auto cby      = Cabana::slice<CBY>(f0)(ii);
            auto dcbydy   = Cabana::slice<DCBYDY>(f0)(ii);
            auto cbz      = Cabana::slice<CBZ>(f0)(ii);
            auto dcbzdz   = Cabana::slice<DCBZDZ>(f0)(ii);

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

            real_t v0 = qdt_4mc/(real_t)sqrt(one + (ux*ux + (uy*uy + uz*uz)));

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

        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );
    Cabana::simd_parallel_for( vec_policy, _uncenter, "uncenter()" );
}

#endif // uncenter
