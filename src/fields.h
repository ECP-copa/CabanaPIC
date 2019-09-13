#ifndef pic_EM_fields_h
#define pic_EM_fields_h

template<class Slice_X, class Slice_Y, class Slice_Z>
KOKKOS_INLINE_FUNCTION
void serial_update_ghosts(
        //field_array_t& fields,
        Slice_X slice_x,
        Slice_Y slice_y,
        Slice_Z slice_z,
        int nx, int ny, int nz, int ng)
{

    const Boundary boundary = Parameters::instance().BOUNDARY_TYPE;
    if (boundary == Boundary::Reflect)
    {
        // TODO: this
        exit(1);
    }
    else { // assume periodic

      // int x,y,z;
      // for ( x = 1; x <= nx; x++){
      // 	//y first
      // 	int to   = VOXEL(x, 1   , 1, nx, ny, nz, ng);
      // 	int from = VOXEL(x, ny+1, 1, nx, ny, nz, ng);
      // 	float tmp_slice_x = slice_x(to);
      // 	//	printf("slice x %d:  %e = %e + %e \n", x, slice_x(to), tmp_slice_x, slice_x(from) );		
      // 	slice_x(to) += slice_x(from);
      // 	slice_x(from) += tmp_slice_x;

	
      // 	to   = VOXEL(x, 1   , 2, nx, ny, nz, ng);
      // 	from = VOXEL(x, ny+1, 2, nx, ny, nz, ng);
      // 	tmp_slice_x = slice_x(to);
      // 	slice_x(to) += slice_x(from);
      // 	slice_x(from) += tmp_slice_x;
      
      // 	//z next
      // 	to   = VOXEL(x, 2, 1   , nx, ny, nz, ng);
      // 	from = VOXEL(x, 2, nz+1, nx, ny, nz, ng);
      // 	tmp_slice_x = slice_x(to);
      // 	slice_x(to) += slice_x(from);
      // 	slice_x(from) += tmp_slice_x;	

      // 	to   = VOXEL(x, 1, 1   , nx, ny, nz, ng);
      // 	from = VOXEL(x, 1, nz+1, nx, ny, nz, ng);
      // 	tmp_slice_x = slice_x(to);
      // 	slice_x(to) += slice_x(from);
      // 	slice_x(from) += tmp_slice_x;	
      // 	//printf("slice x %d, jfx=%e,%e,", x, slice_x(to),slice_x(from));
      // 	to   = VOXEL(x, 2, 1   , nx, ny, nz, ng);
      // 	from = VOXEL(x, 2, nz+1, nx, ny, nz, ng);
      // 	//printf("%e,%e\n",slice_x(to),slice_x(from));	
      // }


      // Copy x from RHS -> LHS
      int x = 1;
      // (1 .. nz+1)
      for (int z = 1; z < nz+2; z++)
      {
          for (int y = 1; y < ny+2; y++)
          {
              // TODO: loop over ng?
              int to = VOXEL(x, y, z, nx, ny, nz, ng);
              int from = VOXEL(nx+1, y, z, nx, ny, nz, ng);

              // Only copy jf? can we copy more values?
              // TODO: once we're in parallel this needs to be a second loop with
              // a buffer
              // Cache value to so we don't lose it during the update
              float tmp_slice_y = slice_y(to);
              float tmp_slice_z = slice_z(to);

              slice_y(to) += slice_y(from);
              slice_z(to) += slice_z(from);
              // printf("slice y %e = %e + %e \n", slice_y(to), tmp_slice_y, slice_y(from) );
              // printf("slice z %e = %e + %e \n", slice_z(to), tmp_slice_z, slice_z(from) );

              // printf("%e + %e = %e \n", tmp_slice_y, slice_y(from), slice_y(to) );

              // TODO: this could just be assignment to slice_y(to)
              slice_y(from) += tmp_slice_y;
              slice_z(from) += tmp_slice_z;

              // TODO: does this copy into the corners twice?
          }
      }

      int y = 1;
      for (int z = 1; z < nz+2; z++)
      {
          for (int x = 1; x < nx+2; x++)
          {
              // TODO: loop over ng?
              int to = VOXEL(x, y, z, nx, ny, nz, ng);
              int from = VOXEL(x, ny+1, z, nx, ny, nz, ng);

              // Only copy jf? can we copy more values?
              // TODO: once we're in parallel this needs to be a second loop with
              // a buffer
              // Cache value to so we don't lose it during the update
              float tmp_slice_x = slice_x(to);
              float tmp_slice_z = slice_z(to);

              slice_x(to) += slice_x(from);
              slice_z(to) += slice_z(from);
              // printf("slice x %e = %e + %e \n", slice_x(to), tmp_slice_x, slice_x(from) );
              // printf("slice z %e = %e + %e \n", slice_z(to), tmp_slice_z, slice_z(from) );

              slice_x(from) += tmp_slice_x;
              slice_z(from) += tmp_slice_z;
          }
      }

      int z = 1;
      for (int y = 1; y < ny+2; y++)
      {
          for (int x = 1; x < nx+2; x++)
          {
              // TODO: loop over ng?
              int to = VOXEL(x, y, z, nx, ny, nz, ng);
              int from = VOXEL(x, y, nz+1, nx, ny, nz, ng);

              // Only copy jf? can we copy more values?
              // TODO: once we're in parallel this needs to be a second loop with
              // a buffer
              // Cache value to so we don't lose it during the update
              float tmp_slice_x = slice_x(to);
              float tmp_slice_y = slice_y(to);

              slice_x(to) += slice_x(from);
              slice_y(to) += slice_y(from);
              // printf("slice x %e = %e + %e \n", slice_x(to), tmp_slice_x, slice_x(from) );
              // printf("slice y %e = %e + %e \n", slice_y(to), tmp_slice_y, slice_y(from) );

              slice_x(from) += tmp_slice_x;
              slice_y(from) += tmp_slice_y;
          }
      }
    }
}
// Policy base class
template<typename Solver_Type> class Field_Solver : public Solver_Type
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
            Solver_Type::advance_b( fields, px, py, pz, nx, ny, nz);
        }
        void advance_e(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		real_t dt_eps0
                )
        {
	  Solver_Type::advance_e( fields, px, py, pz, nx, ny, nz, dt_eps0);
        }
};

// FIXME: Field_solver is repeated => bad naming
class ES_Field_Solver
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
            // No-op, becasue ES
        }

        void advance_e(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		real_t dt_eps0
        )
        {
            auto ex = fields.slice<FIELD_EX>();
            auto ey = fields.slice<FIELD_EY>();
            auto ez = fields.slice<FIELD_EZ>();

            auto cbx = fields.slice<FIELD_CBX>();
            auto cby = fields.slice<FIELD_CBY>();
            auto cbz = fields.slice<FIELD_CBZ>();

            auto jfx = fields.slice<FIELD_JFX>();
            auto jfy = fields.slice<FIELD_JFY>();
            auto jfz = fields.slice<FIELD_JFZ>();

            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
	      const real_t cj =dt_eps0;
                ex(i) = ex(i) + ( - cj * jfx(i) ) ;
                ey(i) = ey(i) + ( - cj * jfy(i) ) ;
                ez(i) = ez(i) + ( - cj * jfz(i) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "es_advance_e()" );
        }
};


class ES_Field_Solver_1D
{
    public:

        real_t e_energy(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz
                )
        {
            auto ex = fields.slice<FIELD_EX>();
            auto ey = fields.slice<FIELD_EY>();
            auto ez = fields.slice<FIELD_EZ>();
            auto _e_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
            {
                lsum += ex(i) * ex(i)
		       +ey(i) * ey(i)
		       +ez(i) * ez(i);
            };

            real_t e_tot_energy=0;
            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_reduce("es_e_energy_1d()", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f;
        }

        void advance_e(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		real_t dt_eps0
        )
        {
            const size_t ng = Parameters::instance().num_ghosts;	  
            auto ex = fields.slice<FIELD_EX>();
            auto ey = fields.slice<FIELD_EY>();
            auto ez = fields.slice<FIELD_EZ>();
            auto jfx = fields.slice<FIELD_JFX>();
            auto jfy = fields.slice<FIELD_JFY>();
            auto jfz = fields.slice<FIELD_JFZ>();
	    
	    serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
	    
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                const real_t cj = dt_eps0;
                ex(i) = ex(i) + ( - cj * jfx(i) ) ;
                ey(i) = ey(i) + ( - cj * jfy(i) ) ;
                ez(i) = ez(i) + ( - cj * jfz(i) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "es_advance_e_1d()" );
        }
};

// EM HERE: UNFINISHED
// TODO: Finish

class EM_Field_Solver
{
    public:

        void advance_b(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		real_t dt_eps0
                )
        {
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


        void advance_e(
                field_array_t fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		real_t dt_eps0
        )
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
               #define UPDATE_EX()                       \
               NEXT_STENCIL();
               }
               */

            auto ex = fields.slice<FIELD_EX>();
            auto ey = fields.slice<FIELD_EY>();
            auto ez = fields.slice<FIELD_EZ>();

            auto cbx = fields.slice<FIELD_CBX>();
            auto cby = fields.slice<FIELD_CBY>();
            auto cbz = fields.slice<FIELD_CBZ>();

            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                real_t old_tcax = 0.0;
                real_t tcax = ( py*(cbz(i)) - pz*(cby(i))) - old_tcax;
                ex(i) = ex(i) +  tcax;

                //f0->tcax = ( py*(f0->cbz*m[f0->fmatz].rmuz-fy->cbz*m[fy->fmatz].rmuz) - pz*(f0->cby*m[f0->fmaty].rmuy-fz->cby*m[fz->fmaty].rmuy) ) - damp*f0->tcax;

                /*
                const float cj   = g->dt/g->eps0;
                f0->ex   = f0->ex + ( - cj*f0->jfx ) ;

                f0->tcay = ( pz*(f0->cbx*m[f0->fmatx].rmux-fz->cbx*m[fz->fmatx].rmux) - px*(f0->cbz*m[f0->fmatz].rmuz-fx->cbz*m[fx->fmatz].rmuz) ) - damp*f0->tcay;
                f0->ey   = m[f0->ematy].decayy*f0->ey + m[f0->ematy].drivey*( f0->tcay - cj*f0->jfy );

                f0->tcaz = ( px*(f0->cby*m[f0->fmaty].rmuy-fx->cby*m[fx->fmaty].rmuy) - py*(f0->cbx*m[f0->fmatx].rmux-fy->cbx*m[fy->fmatx].rmux) ) - damp*f0->tcaz;
                f0->ez   = m[f0->ematz].decayz*f0->ez + m[f0->ematz].drivez*( f0->tcaz - cj*f0->jfz );
                */
                /*
                // UPDATE_EX()
                f0->tcax = ( py*(f0->cbz*m[f0->fmatz].rmuz-fy->cbz*m[fy->fmatz].rmuz) - pz*(f0->cby*m[f0->fmaty].rmuy-fz->cby*m[fz->fmaty].rmuy) ) - damp*f0->tcax;
                f0->ex   = m[f0->ematx].decayx*f0->ex + m[f0->ematx].drivex*( f0->tcax - cj*f0->jfx ) ;

                //#define UPDATE_EY()
                f0->tcay = ( pz*(f0->cbx*m[f0->fmatx].rmux-fz->cbx*m[fz->fmatx].rmux) - px*(f0->cbz*m[f0->fmatz].rmuz-fx->cbz*m[fx->fmatz].rmuz) ) - damp*f0->tcay;
                f0->ey   = m[f0->ematy].decayy*f0->ey + m[f0->ematy].drivey*( f0->tcay - cj*f0->jfy );

                //#define UPDATE_EZ()
                f0->tcaz = ( px*(f0->cby*m[f0->fmaty].rmuy-fx->cby*m[fx->fmaty].rmuy) - py*(f0->cbx*m[f0->fmatx].rmux-fy->cbx*m[fy->fmatx].rmux) ) - damp*f0->tcaz;
                f0->ez   = m[f0->ematz].decayz*f0->ez + m[f0->ematz].drivez*( f0->tcaz - cj*f0->jfz );
                */
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "advance_e()" );

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


#endif // pic_EM_fields_h
