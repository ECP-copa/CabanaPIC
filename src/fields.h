#ifndef pic_fields_h
#define pic_fields_h

class Field_Solver
{
};

class EM_Field_Solver : Field_Solver
{
    public:

        void advance_b(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz
                )
        {
            int n_voxel;

            //f0 = &f(x,  y,  z  );
            //fx = &f(x+1,y,  z  );
            //fy = &f(x,  y+1,z  );
            //fz = &f(x,  y,  z+1);

            auto ex = fields.slice<FIELD_EX>();
            auto ey = fields.slice<FIELD_EY>();
            auto ez = fields.slice<FIELD_EZ>();

            auto cbx = fields.slice<FIELD_CBX>();
            auto cby = fields.slice<FIELD_CBY>();
            auto cbz = fields.slice<FIELD_CBZ>();

            auto _advance_b = KOKKOS_LAMBDA( const int i )
            {
                // Decide if ghost
                bool is_ghost = false;
                // Skip ghosts etc
                if( is_ghost ) { // TODO: should the more common case appear first
                    return;
                }
                //else { // TODO: these is a fairly obvious performance hit to the branching

                size_t f0_index = i; // VOXEL(x,  y,  z,   nx,ny,nz);
                size_t fx_index = i+1; // VOXEL(x+1,y,  z,   nx,ny,nz);
                size_t fy_index = i+(1*nx); // VOXEL(x,  y+1,z,   nx,ny,nz);
                size_t fz_index = i+(1*nx*ny); // VOXEL(x,  y,  z+1, nx,ny,nz);

                // Update value
                /*
                   f0->cbx -= ( py*( fy->ez-f0->ez ) - pz*( fz->ey-f0->ey ) );
                   f0->cby -= ( pz*( fz->ex-f0->ex ) - px*( fx->ez-f0->ez ) );
                   f0->cbz -= ( px*( fx->ey-f0->ey ) - py*( fy->ex-f0->ex ) );
                */
                cbx(i) -= ( py*( ez(fy_index) - ez(f0_index) ) - pz*( ey(fz_index) - ey(f0_index) ) );
                cby(i) -= ( pz*( ex(fz_index) - ex(f0_index) ) - px*( ez(fx_index) - ez(f0_index) ) );
                cbz(i) -= ( px*( ey(fx_index) - ey(f0_index) ) - py*( ex(fy_index) - ex(f0_index) ) );

            };

            // TODO: we could optimize this by cutting off the ghosts from either end
            //Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
            //vec_policy( 0, fields.size() );
            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_b, "advance_b()" );

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
            fz = &f1,ny+1,z+1);
            for( x=1; x<=nx; x++ ) {
            UPDATE_CBY();
            f0++;
            fx++;
            fz
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


        void advance_e()
        {
            // TODO: enable
            /*
               int n_voxel;
               DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
               pipeline_rank, n_pipeline,
               x, y, z, n_voxel );

               INIT_STENCIL();
               for( ; n_voxel; n_voxel-- ) {
               UPDATE_EX(); UPDATE_EY(); UPDATE_EZ();
               NEXT_STENCIL();
               }
               */

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
