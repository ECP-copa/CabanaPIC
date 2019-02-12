#ifndef pic_fields_h
#define pic_fields_h

class EM_Field_Solver
{

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

    void advance_b()
    {
        int n_voxel;

        f0 = &f(x,  y,  z  );
        fx = &f(x+1,y,  z  );
        fy = &f(x,  y+1,z  );
        fz = &f(x,  y,  z+1);

        for( ; n_voxel; n_voxel-- )
        {
            // Update value
            f0->cbx -= ( py*( fy->ez-f0->ez ) - pz*( fz->ey-f0->ey ) );
            f0->cby -= ( pz*( fz->ex-f0->ex ) - px*( fx->ez-f0->ez ) );
            f0->cbz -= ( px*( fx->ey-f0->ey ) - py*( fy->ex-f0->ex ) );

            // Move stencil along
            f0++; fx++; fy++; fz++; x++;
            if( x>nx ) {
                /**/       y++;            x = 1;
                if( y>ny ) z++; if( y>ny ) y = 1;
                INIT_STENCIL();

            }
        }

        /* // TODO: enable this?
           DECLARE_STENCIL();

        // TODO: ask GY how to roll this into the above
        // Do left over bx
        for( z=1; z<=nz; z++ ) {
        for( y=1; y<=ny; y++ ) {
        f0 = &f(nx+1,y,  z);
        fy = &f(nx+1,y+1,z);
        fz = &f(nx+1,y,  z+1);
        UPDATE_CBX();
        }
        }

        // Do left over by
        for( z=1; z<=nz; z++ ) {
        f0 = &f(1,ny+1,z);
        fx = &f(2,ny+1,z);
        fz = &f(1,ny+1,z+1);
        for( x=1; x<=nx; x++ ) {
        UPDATE_CBY();
        f0++;
        fx++;
        fz++;
        }
        }

        // Do left over bz
        for( y=1; y<=ny; y++ ) {
        f0 = &f(1,y,  nz+1);
        fx = &f(2,y,  nz+1);
        fy = &f(1,y+1,nz+1);
        for( x=1; x<=nx; x++ ) {
        UPDATE_CBZ();
        f0++;
        fx++;
        fy++;
        }
        }
        */
    }


    void advance_e_pipeline()
    {
        int n_voxel;
        DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
                pipeline_rank, n_pipeline,
                x, y, z, n_voxel );

        INIT_STENCIL();
        for( ; n_voxel; n_voxel-- ) {
            UPDATE_EX(); UPDATE_EY(); UPDATE_EZ();
            NEXT_STENCIL();
        }

        /* // TODO: enable
           DECLARE_STENCIL();

        // Do left over interior ex
        for( z=2; z<=nz; z++ ) {
        for( y=2; y<=ny; y++ ) {
        f0 = &f(1,y,  z);
        fy = &f(1,y-1,z);
        fz = &f(1,y,  z-1);
        UPDATE_EX();
        }
        }

        // Do left over interior ey
        for( z=2; z<=nz; z++ ) {
        f0 = &f(2,1,z);
        fx = &f(1,1,z);
        fz = &f(2,1,z-1);
        for( x=2; x<=nx; x++ ) {
        UPDATE_EY();
        f0++;
        fx++;
        fz++;
        }
        }

        // Do left over interior ez
        for( y=2; y<=ny; y++ ) {
        f0 = &f(2,y,  1);
        fx = &f(1,y,  1);
        fy = &f(2,y-1,1);
        for( x=2; x<=nx; x++ ) {
        UPDATE_EZ();
        f0++;
        fx++;
        fy++;
        }
        }

        // TODO: this normally neeeds a ghost update..
        */
    }
};

#endif // pic_fields_h
