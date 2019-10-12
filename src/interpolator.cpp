#include "interpolator.h"


void load_interpolator_array(
        field_array_t fields,
        interpolator_array_t interpolators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz,
        size_t ng
        )
{
    size_t x_offset =  1; // VOXEL(x+1,y,  z,   nx,ny,nz);
    size_t y_offset = (1*(nx+ng*2)); // VOXEL(x,  y+1,z,   nx,ny,nz);
    size_t z_offset = (1*(nx+ng*2)*(ny+ng*2)); // VOXEL(x,  y,  z+1, nx,ny,nz);

    auto field_ex = Cabana::slice<FIELD_EX>(fields);
    auto field_ey = Cabana::slice<FIELD_EY>(fields);
    auto field_ez = Cabana::slice<FIELD_EZ>(fields);

    auto field_cbx = Cabana::slice<FIELD_CBX>(fields);
    auto field_cby = Cabana::slice<FIELD_CBY>(fields);
    auto field_cbz = Cabana::slice<FIELD_CBZ>(fields);

    auto interp_ex = Cabana::slice<EX>(interpolators);
    auto interp_dexdy = Cabana::slice<DEXDY>(interpolators);
    auto interp_dexdz = Cabana::slice<DEXDZ>(interpolators);
    auto interp_d2exdydz = Cabana::slice<D2EXDYDZ>(interpolators);
    auto interp_ey = Cabana::slice<EY>(interpolators);
    auto interp_deydz = Cabana::slice<DEYDZ>(interpolators);
    auto interp_deydx = Cabana::slice<DEYDX>(interpolators);
    auto interp_d2eydzdx = Cabana::slice<D2EYDZDX>(interpolators);
    auto interp_ez = Cabana::slice<EZ>(interpolators);
    auto interp_dezdx = Cabana::slice<DEZDX>(interpolators);
    auto interp_dezdy = Cabana::slice<DEZDY>(interpolators);
    auto interp_d2ezdxdy = Cabana::slice<D2EZDXDY>(interpolators);
    auto interp_cbx = Cabana::slice<CBX>(interpolators);
    auto interp_dcbxdx = Cabana::slice<DCBXDX>(interpolators);
    auto interp_cby = Cabana::slice<CBY>(interpolators);
    auto interp_dcbydy = Cabana::slice<DCBYDY>(interpolators);
    auto interp_cbz = Cabana::slice<CBZ>(interpolators);
    auto interp_dcbzdz = Cabana::slice<DCBZDZ>(interpolators);

    const real_t fourth = 1.0 / 4.0;
    const real_t half = 1.0 / 2.0;
    
    // TODO: we have to be careful we don't reach past the ghosts here
    auto _load_interpolator = KOKKOS_LAMBDA( const int x, const int y, const int z)
    {
        // Try avoid doing stencil operations on ghost cells
        //if ( is_ghost(i) ) continue;

        int i = VOXEL(x,y,z, nx,ny,nz,ng);

        // ex interpolation
        real_t w0 = field_ex(i);                       // pf0->ex;
        real_t w1 = field_ex(i + y_offset);            // pfy->ex;
        real_t w2 = field_ex(i + z_offset);            // pfz->ex;
        real_t w3 = field_ex(i + y_offset + z_offset); // pfyz->ex;

        // TODO: make this not use only w0
        interp_ex(i)       = fourth*( (w3 + w0) + (w1 + w2) );
        interp_dexdy(i)    = fourth*( (w3 - w0) + (w1 - w2) );
        interp_dexdz(i)    = fourth*( (w3 - w0) - (w1 - w2) );
        interp_d2exdydz(i) = fourth*( (w3 + w0) - (w1 + w2) );

        // ey interpolation coefficients
        w0 = field_ey(i);
        w1 = field_ey(i + z_offset); // pfz->ey;
        w2 = field_ey(i + x_offset); //pfx->ey;
        w3 = field_ey(i + x_offset + z_offset); // pfzx->ey;

        interp_ey(i)       = fourth*( (w3 + w0) + (w1 + w2) );
        interp_deydz(i)    = fourth*( (w3 - w0) + (w1 - w2) );
        interp_deydx(i)    = fourth*( (w3 - w0) - (w1 - w2) );
        interp_d2eydzdx(i) = fourth*( (w3 + w0) - (w1 + w2) );


        // ez interpolation coefficients
        w0 = field_ez(i); // pf0->ez;
        w1 = field_ez(i + x_offset); //pfx->ez;
        w2 = field_ez(i + y_offset); //pfy->ez;
        w3 = field_ez(i + x_offset + y_offset); //pfxy->ez;

        interp_ez(i)       = fourth*( (w3 + w0) + (w1 + w2) );
        interp_dezdx(i)    = fourth*( (w3 - w0) + (w1 - w2) );
        interp_dezdy(i)    = fourth*( (w3 - w0) - (w1 - w2) );
        interp_d2ezdxdy(i) = fourth*( (w3 + w0) - (w1 + w2) );

        // bx interpolation coefficients
        w0 = field_cbx(i); //pf0->cbx;
        w1 = field_cbx(i + x_offset); //pfx->cbx;
        interp_cbx(i)    = half*( w1 + w0 );
        interp_dcbxdx(i) = half*( w1 - w0 );

        // by interpolation coefficients
        w0 = field_cby(i); // pf0->cby;
        w1 = field_cby(i + y_offset); // pfy->cby;
        interp_cby(i)    = half*( w1 + w0 );
        interp_dcbydy(i) = half*( w1 - w0 );

        // bz interpolation coefficients
        w0 = field_cbz(i); // pf0->cbz;
        w1 = field_cbz(i + z_offset); // pfz->cbz;
        interp_cbz(i)    = half*( w1 + w0 );
        interp_dcbzdz(i) = half*( w1 - w0 );
    };

    //Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() ); // All cells
    Kokkos::MDRangePolicy< Kokkos::Rank<3> > non_ghost_policy( {ng,ng,ng}, {nx+ng, ny+ng, nz+ng} ); // Try not to into ghosts // TODO: dry this
    Kokkos::parallel_for( non_ghost_policy, _load_interpolator, "load_interpolator()" );

        /*
        pi   = &fi(x,  y,  z  );
        pf0  =  &f(x,  y,  z  );
        pfx  =  &f(x+1,y,  z  );
        pfy  =  &f(x,  y+1,z  );
        pfz  =  &f(x,  y,  z+1);
        pfyz =  &f(x,  y+1,z+1);
        pfzx =  &f(x+1,y,  z+1);
        pfxy =  &f(x+1,y+1,z  );
        */

}
void initialize_interpolator(interpolator_array_t& f0)
{
    auto ex = Cabana::slice<EX>(f0);
    auto dexdy  = Cabana::slice<DEXDY>(f0);
    auto dexdz  = Cabana::slice<DEXDZ>(f0);
    auto d2exdydz  = Cabana::slice<D2EXDYDZ>(f0);
    auto ey  = Cabana::slice<EY>(f0);
    auto deydz  = Cabana::slice<DEYDZ>(f0);
    auto deydx  = Cabana::slice<DEYDX>(f0);
    auto d2eydzdx  = Cabana::slice<D2EYDZDX>(f0);
    auto ez  = Cabana::slice<EZ>(f0);
    auto dezdx  = Cabana::slice<DEZDX>(f0);
    auto dezdy  = Cabana::slice<DEZDY>(f0);
    auto d2ezdxdy  = Cabana::slice<D2EZDXDY>(f0);
    auto cbx  = Cabana::slice<CBX>(f0);
    auto dcbxdx   = Cabana::slice<DCBXDX>(f0);
    auto cby  = Cabana::slice<CBY>(f0);
    auto dcbydy  = Cabana::slice<DCBYDY>(f0);
    auto cbz  = Cabana::slice<CBZ>(f0);
    auto dcbzdz  = Cabana::slice<DCBZDZ>(f0);

    auto _init_interpolator =
        KOKKOS_LAMBDA( const int i )
        {
            // Throw in some place holder values
            ex(i) = 0.0;
            dexdy(i) = 0.0;
            dexdz(i) = 0.0;
            d2exdydz(i) = 0.0;
            ey(i) = 0.0;
            deydz(i) = 0.0;
            deydx(i) = 0.0;
            d2eydzdx(i) = 0.0;
            ez(i) = 0.0;
            dezdx(i) = 0.0;
            dezdy(i) = 0.0;
            d2ezdxdy(i) = 0.0;
            cbx(i) = 0.0;
            dcbxdx(i) = 0.0;
            cby(i) = 0.0;
            dcbydy(i) = 0.0;
            cbz(i) = 0.0;
            dcbzdz(i) = 0.0;
        };

    Kokkos::parallel_for( f0.size(), _init_interpolator, "init_interpolator()" );

}
