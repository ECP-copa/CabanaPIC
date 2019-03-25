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
    size_t x_offset = 1; // VOXEL(x+1,y,  z,   nx,ny,nz);
    size_t y_offset = (1*nx); // VOXEL(x,  y+1,z,   nx,ny,nz);
    size_t z_offset = (1*nx*ny); // VOXEL(x,  y,  z+1, nx,ny,nz);

    auto field_ex = fields.slice<FIELD_EX>();
    auto field_ey = fields.slice<FIELD_EY>();
    auto field_ez = fields.slice<FIELD_EZ>();

    auto field_cbx = fields.slice<FIELD_CBX>();
    auto field_cby = fields.slice<FIELD_CBY>();
    auto field_cbz = fields.slice<FIELD_CBZ>();

    auto interp_ex = interpolators.slice<EX>();
    auto interp_dexdy = interpolators.slice<DEXDY>();
    auto interp_dexdz = interpolators.slice<DEXDZ>();
    auto interp_d2exdydz = interpolators.slice<D2EXDYDZ>();
    auto interp_ey = interpolators.slice<EY>();
    auto interp_deydz = interpolators.slice<DEYDZ>();
    auto interp_deydx = interpolators.slice<DEYDX>();
    auto interp_d2eydzdx = interpolators.slice<D2EYDZDX>();
    auto interp_ez = interpolators.slice<EZ>();
    auto interp_dezdx = interpolators.slice<DEZDX>();
    auto interp_dezdy = interpolators.slice<DEZDY>();
    auto interp_d2ezdxdy = interpolators.slice<D2EZDXDY>();
    auto interp_cbx = interpolators.slice<CBX>();
    auto interp_dcbxdx = interpolators.slice<DCBXDX>();
    auto interp_cby = interpolators.slice<CBY>();
    auto interp_dcbydy = interpolators.slice<DCBYDY>();
    auto interp_cbz = interpolators.slice<CBZ>();
    auto interp_dcbzdz = interpolators.slice<DCBZDZ>();

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

        //1D only
        // TODO: make this not use only w0
        interp_ex(i)       = w0; //fourth*( (w3 + w0) + (w1 + w2) );
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
