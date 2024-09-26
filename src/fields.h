#ifndef pic_EM_fields_h
#define pic_EM_fields_h

#include <fstream>
#include "Cabana_Parallel.hpp" // Simd parallel for
#include "Cabana_DeepCopy.hpp" // Cabana::deep_copy
#include "input/deck.h"

#ifdef USE_GPU
#include <cufft.h>
#else
#include "fftw3.h"
#endif

// TODO: Namespace this stuff?
template<class Slice_X, class Slice_Y, class Slice_Z>
void serial_update_ghosts_B(
        //field_array_t& fields,
        Slice_X slice_x,
        Slice_Y slice_y,
        Slice_Z slice_z,
        int nx, int ny, int nz, int ng)
{
    // assume periodic

        // TODO: it may be worth turning these into fewer kernels, as they
        // really don't have a lot of work

        //for (int z = 1; z < nz+1; z++) {
            //for (int y = 1; y < ny+1; y++) {
        auto _zy_boundary = KOKKOS_LAMBDA( const int z, const int y )
        {
            // Copy x from LHS -> RHS
            int from = VOXEL(1   , y, z, nx, ny, nz, ng);
            int to   = VOXEL(nx+1, y, z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);

            // Copy x from RHS -> LHS
            from = VOXEL(nx  , y, z, nx, ny, nz, ng);
            to   = VOXEL(0   , y, z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);
        };
        //}

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1,1}, {nz+1,ny+1});
        Kokkos::parallel_for( "zy boundary()", zy_policy, _zy_boundary );

        //for (int x = 0; x < nx+2; x++) {
            //for (int z = 1; z < nz+1; z++) {
        auto _xz_boundary = KOKKOS_LAMBDA( const int x, const int z )
        {
            int from = VOXEL(x,    1, z, nx, ny, nz, ng);
            int to   = VOXEL(x, ny+1, z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);

            from = VOXEL(x, ny  , z, nx, ny, nz, ng);
            to   = VOXEL(x, 0   , z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);
        };
        //}
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_policy({0,1}, {nx+2,nz+1});
        Kokkos::parallel_for( "xz boundary()", xz_policy, _xz_boundary );

        //for (int y = 0; y < ny+2; y++) {
            //for (int x = 0; x < nx+2; x++) {
        auto _yx_boundary = KOKKOS_LAMBDA( const int y, const int x )
        {
            int from = VOXEL(x, y, 1   , nx, ny, nz, ng);
            int to   = VOXEL(x, y, nz+1, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);

            from = VOXEL(x, y, nz  , nx, ny, nz, ng);
            to   = VOXEL(x, y, 0   , nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);
        };
        //}
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({0,0}, {ny+2,nx+2});
        Kokkos::parallel_for( "yx boundary()", yx_policy, _yx_boundary );

}

template<class Slice_X, class Slice_Y, class Slice_Z>
//KOKKOS_INLINE_FUNCTION
void serial_update_ghosts(
        //field_array_t& fields,
        Slice_X slice_x,
        Slice_Y slice_y,
        Slice_Z slice_z,
        int nx, int ny, int nz, int ng)
{

 // assume periodic

        /*
           To fill in contributions from places of periodic BC
           */
        //TODO: theses again don't have a sensible amount of work

        //for ( x = 1; x <= nx; x++ ){
        auto _x_boundary = KOKKOS_LAMBDA( const int x )
        {
            for(int z = 1; z <= nz+1; z++){
                //y first
                int from = VOXEL(x, ny+1, z, nx, ny, nz, ng);
                int to   = VOXEL(x, 1   , z, nx, ny, nz, ng);
                slice_x(to) += slice_x(from);
            }

            for(int y = 1; y <= ny+1; y++){
                //z next
                int from = VOXEL(x, y, nz+1, nx, ny, nz, ng);
                int to   = VOXEL(x, y, 1   , nx, ny, nz, ng);
                slice_x(to) += slice_x(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> x_policy(1, nx+1);
        Kokkos::parallel_for( "_x_boundary()", x_policy, _x_boundary );

        //for ( y = 1; y <= ny; y++ ){
        auto _y_boundary = KOKKOS_LAMBDA( const int y )
        {
            for (int x = 1; x <= nx+1; x++){
                //z first
                int from = VOXEL(x   , y, nz+1, nx, ny, nz, ng);
                int to   = VOXEL(x   , y, 1   , nx, ny, nz, ng);
                slice_y(to) += slice_y(from);
            }

            for (int z = 1; z <= nz+1; z++){
                //x next
                int from = VOXEL(nx+1, y, z   , nx, ny, nz, ng);
                int to   = VOXEL(1   , y, z   , nx, ny, nz, ng);
                slice_y(to) += slice_y(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> y_policy(1, ny+1);
        Kokkos::parallel_for( "_y_boundary()", y_policy, _y_boundary );

        //for ( z = 1; z <= nz; z++ ){
        auto _z_boundary = KOKKOS_LAMBDA( const int z )
        {
            for (int y = 1; y <= ny+1; y++){
                //x first
                int from = VOXEL(nx+1, y   , z, nx, ny, nz, ng);
                int to   = VOXEL(1   , y   , z, nx, ny, nz, ng);
                slice_z(to) += slice_z(from);
            }

            for (int x = 1; x <= nx+1; x++){
                //y next
                int from = VOXEL(x   , ny+1, z, nx, ny, nz, ng);
                int to   = VOXEL(x   , 1   , z, nx, ny, nz, ng);
                slice_z(to) += slice_z(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> z_policy(1, nz+1);
        Kokkos::parallel_for( "_z_boundary()", z_policy, _z_boundary );

        // // Copy x from RHS -> LHS
        // int x = 1;
        // // (1 .. nz+1)
        // for (int z = 1; z < nz+2; z++)
        //   {
        // 	for (int y = 1; y < ny+2; y++)
        //       {
        // 	    // TODO: loop over ng?
        // 	    int to = VOXEL(x, y, z, nx, ny, nz, ng);
        // 	    int from = VOXEL(nx+1, y, z, nx, ny, nz, ng);

        // 	    // Only copy jf? can we copy more values?
        // 	    // TODO: once we're in parallel this needs to be a second loop with
        // 	    // a buffer
        // 	    // Cache value to so we don't lose it during the update
        // 	    tmp_slice_y = slice_y(to);
        // 	    tmp_slice_z = slice_z(to);

        // 	    slice_y(to) += slice_y(from);
        // 	    slice_z(to) += slice_z(from);
        // 	    // printf("slice y %e = %e + %e \n", slice_y(to), tmp_slice_y, slice_y(from) );
        // 	    // printf("slice z %e = %e + %e \n", slice_z(to), tmp_slice_z, slice_z(from) );

        // 	    // printf("%e + %e = %e \n", tmp_slice_y, slice_y(from), slice_y(to) );

        // 	    // TODO: this could just be assignment to slice_y(to)
        // 	    slice_y(from) += tmp_slice_y;
        // 	    slice_z(from) += tmp_slice_z;

        // 	    // TODO: does this copy into the corners twice?
        //       }
        //   }

        // int y = 1;
        // for (int z = 1; z < nz+2; z++)
        //   {
        // 	for (int x = 1; x < nx+2; x++)
        //       {
        // 	    // TODO: loop over ng?
        // 	    int to = VOXEL(x, y, z, nx, ny, nz, ng);
        // 	    int from = VOXEL(x, ny+1, z, nx, ny, nz, ng);

        // 	    // Only copy jf? can we copy more values?
        // 	    // TODO: once we're in parallel this needs to be a second loop with
        // 	    // a buffer
        // 	    // Cache value to so we don't lose it during the update
        // 	    tmp_slice_x = slice_x(to);
        // 	    tmp_slice_z = slice_z(to);

        // 	    slice_x(to) += slice_x(from);
        // 	    slice_z(to) += slice_z(from);
        // 	    // printf("slice x %e = %e + %e \n", slice_x(to), tmp_slice_x, slice_x(from) );
        // 	    // printf("slice z %e = %e + %e \n", slice_z(to), tmp_slice_z, slice_z(from) );

        // 	    slice_x(from) += tmp_slice_x;
        // 	    slice_z(from) += tmp_slice_z;
        //       }
        //   }

        // int z = 1;
        // for (int y = 1; y < ny+2; y++)
        //   {
        // 	for (int x = 1; x < nx+2; x++)
        //       {
        // 	    // TODO: loop over ng?
        // 	    int to = VOXEL(x, y, z, nx, ny, nz, ng);
        // 	    int from = VOXEL(x, y, nz+1, nx, ny, nz, ng);

        // 	    // Only copy jf? can we copy more values?
        // 	    // TODO: once we're in parallel this needs to be a second loop with
        // 	    // a buffer
        // 	    // Cache value to so we don't lose it during the update
        // 	    tmp_slice_x = slice_x(to);
        // 	    tmp_slice_y = slice_y(to);

        // 	    slice_x(to) += slice_x(from);
        // 	    slice_y(to) += slice_y(from);
        // 	    // printf("slice x %e = %e + %e \n", slice_x(to), tmp_slice_x, slice_x(from) );
        // 	    // printf("slice y %e = %e + %e \n", slice_y(to), tmp_slice_y, slice_y(from) );

        // 	    slice_x(from) += tmp_slice_x;
        // 	    slice_y(from) += tmp_slice_y;
        //       }
        //   }

}


// Policy base class
template<typename Solver_Type> class Field_Solver : public Solver_Type
{
    public:

        //constructor
        Field_Solver(field_array_t& fields)
        {
            // Zero the fields so everything has a safe value.
            // This occurs before we parse any custom fields in a user deck
            init(fields);
        }

        void init(field_array_t& fields)
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

            auto _init_fields =
                KOKKOS_LAMBDA( const int i )
                {
                    ex(i) = 0.0;
                    ey(i) = 0.0;
                    ez(i) = 0.0;
                    cbx(i) = 0.0;
                    cby(i) = 0.0;
                    cbz(i) = 0.0;
                    jfx(i) = 0.0;
                    jfy(i) = 0.0;
                    jfz(i) = 0.0;
                };

            Kokkos::parallel_for( "init_fields()", fields.size(), _init_fields );
        }

        // TODO: is this the right place for this vs in the helper?
        void dump_fields(FILE * fp,
                field_array_t& d_fields,
                real_t xmin,
                real_t,
                real_t,
                real_t dx,
                real_t,
                real_t,
                size_t nx,
                size_t ny,
                size_t,
                size_t ng
                )
        {
            // Host
            field_array_t::host_mirror_type fields("host_fields", d_fields.size());

            // Copy device field to host
            Cabana::deep_copy(fields, d_fields);

            auto ex = Cabana::slice<FIELD_EX>(fields);

            for( size_t i=1; i<nx+1; i++ )
            {
                real_t x = xmin + (i-0.5)*dx;
                size_t ii = VOXEL(i,1,1,nx,ny,nz,ng);
                //	  fprintf(fp,"%e %e %e %e %e %e %e\n",x,y,ey(ii),jfx(ii),jfy(ii),jfz(ii),cbz(ii));
                fprintf(fp,"%e %e\n",x,ex(ii));
            }

            fprintf(fp,"\n\n");

        }

        void advance_b(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            Solver_Type::advance_b( fields, px, py, pz, nx, ny, nz, ng);
        }
        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            Solver_Type::advance_e( fields, px, py, pz, nx, ny, nz, ng, dt_eps0);
        }
};

// FIXME: Field_solver is repeated => bad naming
class ES_Field_Solver
{
    public:

        void advance_b(
                field_array_t&,
                real_t,
                real_t,
                real_t,
                size_t,
                size_t,
                size_t,
                size_t
                )
        {
            // No-op, becasue ES
        }

        void advance_e(
                field_array_t& fields,
                real_t,
                real_t,
                real_t,
                size_t,
                size_t,
                size_t,
                size_t,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

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
            Kokkos::parallel_for( "es_advance_e()", exec_policy, _advance_e );
        }

        real_t e_energy(
                field_array_t& fields,
                real_t,
                real_t,
                real_t,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
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
};


class ES_Field_Solver_Spectral
{
    public:
        void advance_b(
                field_array_t&,
                real_t,
                real_t,
                real_t,
                size_t,
                size_t,
                size_t,
                size_t
                )
        {
            // No-op, becasue ES
        }

        real_t e_energy(
                field_array_t& fields,
                real_t dx,
                real_t dy,
                real_t dz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto _e_energy =  KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum ) // KOKKOS_LAMBDA( const int i, real_t & lsum )
            {
	     const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
	     lsum += ex(i) * ex(i)
	     +ey(i) * ey(i)
	     +ez(i) * ez(i);
	     //printf("%d, %d, %d, %d %e\n",x,y,z,i,ex(i));
            };

            real_t e_tot_energy=0;
	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});
	    Kokkos::parallel_reduce("e_energy", fft_exec_policy, _e_energy, e_tot_energy );
            //Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            //Kokkos::parallel_reduce("es_e_energy_1d()", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f*dx*dy*dz;
        }

        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

	    serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);

	    size_t n_inner_cell = nx*ny*nz;
	    real_t iLx = px/(nx*dt_eps0);
	    real_t iLy = py/(ny*dt_eps0);
	    real_t iLz = pz/(nz*dt_eps0);

	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});

	    //for many ffts
	    int rank = 3;
	    int n[3] = {nx, ny, nz};
	    int depth = 3;
	    int idist = 1, odist = 1;
	    int istride = 3,ostride = 3;
	    int *inembed = n, *onembed = n;

	    ViewVecComplex fft_coefficients("fft_coef", n_inner_cell*3);
	    ViewVecComplex fft_out("fft_out", n_inner_cell*3);
	    
	    //start fft
#ifdef USE_GPU
	    // CUFFT plans
	    cufftHandle forward_plan, inverse_plan;
	    cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, depth);

	    cufftPlanMany(&inverse_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, depth);
#else
	    	    //plan
	    fftwf_plan plan_fft = fftwf_plan_many_dft (rank, //rank
						       n, //dims -- this doesn't include zero-padding
						       depth, //howmany
						       reinterpret_cast<fftwf_complex *>(fft_coefficients.data()), //in
						       inembed, //inembed
						       depth, //istride
						       idist, //idist
						       reinterpret_cast<fftwf_complex *>(fft_out.data()), //out
						       onembed, //onembed
						       depth, //ostride
						       odist, //odist
						       FFTW_FORWARD,  
						       FFTW_ESTIMATE );
	    
	    fftwf_plan plan_ifft = fftwf_plan_many_dft (rank, //rank
						       n, //dims -- this doesn't include zero-padding
						       depth, //howmany
							reinterpret_cast<fftwf_complex *>(fft_out.data()), //in
						       inembed, //inembed
						       depth, //istride
						       idist, //idist
							reinterpret_cast<fftwf_complex *>(fft_coefficients.data()), //out
						       onembed, //onembed
						       depth, //ostride
						       odist, //odist
						       FFTW_BACKWARD,  
						       FFTW_ESTIMATE );
#endif

	    auto _find_jf_fft = KOKKOS_LAMBDA( const int i, const int j, const int k  ){ //for(int i=0; i<ny; ++i){
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
		const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
		fft_coefficients(3*f0+0) = jfx(f1);
		fft_coefficients(3*f0+1) = jfy(f1);
		fft_coefficients(3*f0+2) = jfz(f1);

		//printf ("real part = %15.8e  imag part = %15.8e\n", fft_coefficients[i].x, fft_coefficients[i].y);
	    };   

	    Kokkos::parallel_for("find_jf_fft", fft_exec_policy, _find_jf_fft );

#ifdef USE_GPU	    
	    // Transform signal 
	    cufftExecC2C(forward_plan, (cufftComplex *)(fft_coefficients.data()), (cufftComplex *)(fft_out.data()), CUFFT_FORWARD);
#else
	    fftwf_execute(plan_fft);
#endif	    
	    //Perform div,poisson solve, and gradient in Fourier space
	    auto _DPG_Fourier = KOKKOS_LAMBDA(  const int i, const int j, const int k ){ 
		int i0 = i-ng;
		int j0 = j-ng;
		int k0 = k-ng;

		real_t kx,ky,kz;
		const int f0 = VOXEL(i0,   j0,   k0,   nx, ny, nz, 0);
		//d(jfx)/dx
		if(i0<(nx+1)/2) {
		    kx = (real_t)i0*2.0*M_PI*iLx;
		}else{
		    kx = (real_t)(nx-i0)*2.0*M_PI*iLx;
		}

		//d(jfy)/dy
		if(j0<(ny+1)/2) {
		    ky = (real_t)j0*2.0*M_PI*iLy;
		}else{
		    ky = (real_t)(ny-j0)*2.0*M_PI*iLy;
		}

		//d(jfz)/dz
		if(k0<(nz+1)/2) {
		    kz = (real_t)k0*2.0*M_PI*iLz;
		}else{
		    kz = (real_t)(nz-k0)*2.0*M_PI*iLz;
		}
		
		Kokkos::complex<real_t> kdotj = fft_out(3*f0+0)*kx+fft_out(3*f0+1)*ky+fft_out(3*f0+2)*kz;
		real_t k2 = kx*kx+ky*ky+kz*kz;
		if(i0==0&&j0==0&&k0==0) k2 = 1.;
		Kokkos::complex<real_t> phif = kdotj/k2;
		if(i0==0&&j0==0&&k0==0) phif = 0.;
		fft_out(3*f0+0)=kx*phif;
		fft_out(3*f0+1)=ky*phif;
		fft_out(3*f0+2)=kz*phif;
	    };

	    Kokkos::parallel_for("DPG_Fourier", fft_exec_policy, _DPG_Fourier );

#ifdef USE_GPU
	    // Transform signal back
	    cufftExecC2C(inverse_plan, (cufftComplex *)(fft_out.data()), (cufftComplex *)(fft_coefficients.data()), CUFFT_INVERSE);
	    cufftDestroy(inverse_plan);
	    cufftDestroy(forward_plan);
#else
	    fftwf_execute(plan_ifft);
	    fftwf_destroy_plan(plan_fft);
	    fftwf_destroy_plan(plan_ifft);
#endif

	    auto _find_jf_ifft = KOKKOS_LAMBDA( const int i, const int j, const int k  ){ //for(int i=0; i<ny; ++i){
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
	    	const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
	    	jfx(f1) = fft_coefficients(3*f0+0).real()/n_inner_cell;
	    	jfy(f1) = fft_coefficients(3*f0+1).real()/n_inner_cell;
	    	jfz(f1) = fft_coefficients(3*f0+2).real()/n_inner_cell;
	    };

	    Kokkos::parallel_for("find_jf_ifft", fft_exec_policy, _find_jf_ifft );	    
	    //end fft
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);



	    //remove the average (for 1D problems only)
	    real_t jx_avg = 0,jy_avg=0,jz_avg=0;
	    
	    if(nx==1||ny==1||nz==1){
		if(nx>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfx(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jx_avg );
		    jx_avg /=n_inner_cell;
		}

		if(ny>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfy(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jy_avg );
		    jy_avg /=n_inner_cell;
		}

		if(nz>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfz(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jz_avg );
		    jz_avg /=n_inner_cell;
		}
	    }
	    
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
	    const real_t cjx = (nx>1)?dt_eps0:0;
	    const real_t cjy = (ny>1)?dt_eps0:0;
	    const real_t cjz = (nz>1)?dt_eps0:0;
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                ex(i) = ex(i) + ( - cjx * (jfx(i)-jx_avg) ) ;
                ey(i) = ey(i) + ( - cjy * (jfy(i)-jy_avg) ) ;
                ez(i) = ez(i) + ( - cjz * (jfz(i)-jz_avg) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for(  "es_advance_e()", exec_policy, _advance_e );
        }

    
    	  // Given E_n and E_{n+1/2}, puts E_{n+1} into the array of E_{n+1/2}

        void extend_e(
		  field_array_t& fields_nph, 
		  field_array_t& fields_n)
    {
            auto ex = Cabana::slice<FIELD_EX>(fields_nph);
            auto ey = Cabana::slice<FIELD_EY>(fields_nph);
            auto ez = Cabana::slice<FIELD_EZ>(fields_nph);
            
            auto ex_old = Cabana::slice<FIELD_EX>(fields_n);
            auto ey_old = Cabana::slice<FIELD_EY>(fields_n);
            auto ez_old = Cabana::slice<FIELD_EZ>(fields_n);
				
	    auto _extend_e = KOKKOS_LAMBDA( const int i )
            {
                ex(i) = ex(i)*2.0  - ex_old(i)  ;
                ey(i) = ey(i)*2.0  - ey_old(i)  ;
                ez(i) = ez(i)*2.0  - ez_old(i)  ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields_nph.size() );
            Kokkos::parallel_for( "extend_e()", exec_policy, _extend_e );
    }

};


class ES_Field_Solver_1D
{
    public:
        void advance_b(
                field_array_t&,
                real_t,
                real_t,
                real_t,
                size_t,
                size_t,
                size_t,
                size_t
                )
        {
            // No-op, becasue ES
        }

        real_t e_energy(
                field_array_t& fields,
                real_t dx,
                real_t dy,
                real_t dz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto _e_energy =  KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum ) // KOKKOS_LAMBDA( const int i, real_t & lsum )
            {
	     const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
	     lsum += ex(i) * ex(i)
	     +ey(i) * ey(i)
	     +ez(i) * ez(i);
	     //printf("xyz=%d,%d,%d, exyz=%e,%e,%e\n",x,y,z,ex(i),ey(i),ez(i));
            };
            real_t e_tot_energy=0;
	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});
	    Kokkos::parallel_reduce("e_energy", fft_exec_policy, _e_energy, e_tot_energy );
            //Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            //Kokkos::parallel_reduce("es_e_energy_1d()", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f*dx*dy*dz;
        }
    /*
        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

	    serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);

	    size_t n_inner_cell = nx*ny*nz;
	    real_t iLx = px/(nx*dt_eps0);
	    real_t iLy = py/(ny*dt_eps0);
	    real_t iLz = pz/(nz*dt_eps0);

	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});

	    //for many ffts
	    int rank = 3;
	    int n[3] = {nx, ny, nz};
	    int depth = 3;
	    int idist = 1, odist = 1;
	    int istride = 3,ostride = 3;
	    int *inembed = n, *onembed = n;

	    ViewVecComplex fft_coefficients("fft_coef", n_inner_cell*3);
	    ViewVecComplex fft_out("fft_out", n_inner_cell*3);
	    
	    //start fft
#ifdef USE_GPU
	    // CUFFT plans
	    cufftHandle forward_plan, inverse_plan;
	    cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, depth);

	    cufftPlanMany(&inverse_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, depth);
#else
	    	    //plan
	    fftwf_plan plan_fft = fftwf_plan_many_dft (rank, //rank
						       n, //dims -- this doesn't include zero-padding
						       depth, //howmany
						       reinterpret_cast<fftwf_complex *>(fft_coefficients.data()), //in
						       inembed, //inembed
						       depth, //istride
						       idist, //idist
						       reinterpret_cast<fftwf_complex *>(fft_out.data()), //out
						       onembed, //onembed
						       depth, //ostride
						       odist, //odist
						       FFTW_FORWARD,  
						       FFTW_ESTIMATE );
	    
	    fftwf_plan plan_ifft = fftwf_plan_many_dft (rank, //rank
						       n, //dims -- this doesn't include zero-padding
						       depth, //howmany
							reinterpret_cast<fftwf_complex *>(fft_out.data()), //in
						       inembed, //inembed
						       depth, //istride
						       idist, //idist
							reinterpret_cast<fftwf_complex *>(fft_coefficients.data()), //out
						       onembed, //onembed
						       depth, //ostride
						       odist, //odist
						       FFTW_BACKWARD,  
						       FFTW_ESTIMATE );
#endif

	    auto _find_jf_fft = KOKKOS_LAMBDA( const int i, const int j, const int k  ){ //for(int i=0; i<ny; ++i){
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
		const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
		fft_coefficients(3*f0+0) = jfx(f1);
		fft_coefficients(3*f0+1) = jfy(f1);
		fft_coefficients(3*f0+2) = jfz(f1);

		//printf ("real part = %15.8e  imag part = %15.8e\n", fft_coefficients[i].x, fft_coefficients[i].y);
	    };   

	    Kokkos::parallel_for("find_jf_fft", fft_exec_policy, _find_jf_fft );

#ifdef USE_GPU	    
	    // Transform signal 
	    cufftExecC2C(forward_plan, (cufftComplex *)(fft_coefficients.data()), (cufftComplex *)(fft_out.data()), CUFFT_FORWARD);
#else
	    fftwf_execute(plan_fft);
#endif	    
	    //Perform div,poisson solve, and gradient in Fourier space
	    auto _DPG_Fourier = KOKKOS_LAMBDA(  const int i, const int j, const int k ){ 
		int i0 = i-ng;
		int j0 = j-ng;
		int k0 = k-ng;

		real_t kx,ky,kz;
		const int f0 = VOXEL(i0,   j0,   k0,   nx, ny, nz, 0);
		//d(jfx)/dx
		if(i0<(nx+1)/2) {
		    kx = (real_t)i0*2.0*M_PI*iLx;
		}else{
		    kx = (real_t)(nx-i0)*2.0*M_PI*iLx;
		}

		//d(jfy)/dy
		if(j0<(ny+1)/2) {
		    ky = (real_t)j0*2.0*M_PI*iLy;
		}else{
		    ky = (real_t)(ny-j0)*2.0*M_PI*iLy;
		}

		//d(jfz)/dz
		if(k0<(nz+1)/2) {
		    kz = (real_t)k0*2.0*M_PI*iLz;
		}else{
		    kz = (real_t)(nz-k0)*2.0*M_PI*iLz;
		}
		
		Kokkos::complex<real_t> kdotj = fft_out(3*f0+0)*kx+fft_out(3*f0+1)*ky+fft_out(3*f0+2)*kz;
		real_t k2 = kx*kx+ky*ky+kz*kz;
		if(i0==0&&j0==0&&k0==0) k2 = 1.;
		Kokkos::complex<real_t> phif = kdotj/k2;
		if(i0==0&&j0==0&&k0==0) phif = 0.;
		fft_out(3*f0+0)=kx*phif;
		fft_out(3*f0+1)=ky*phif;
		fft_out(3*f0+2)=kz*phif;
	    };

	    Kokkos::parallel_for("DPG_Fourier", fft_exec_policy, _DPG_Fourier );

#ifdef USE_GPU
	    // Transform signal back
	    cufftExecC2C(inverse_plan, (cufftComplex *)(fft_out.data()), (cufftComplex *)(fft_coefficients.data()), CUFFT_INVERSE);
	    cufftDestroy(inverse_plan);
	    cufftDestroy(forward_plan);
#else
	    fftwf_execute(plan_ifft);
	    fftwf_destroy_plan(plan_fft);
	    fftwf_destroy_plan(plan_ifft);
#endif

	    auto _find_jf_ifft = KOKKOS_LAMBDA( const int i, const int j, const int k  ){ //for(int i=0; i<ny; ++i){
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
	    	const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
	    	jfx(f1) = fft_coefficients(3*f0+0).real()/n_inner_cell;
	    	jfy(f1) = fft_coefficients(3*f0+1).real()/n_inner_cell;
	    	jfz(f1) = fft_coefficients(3*f0+2).real()/n_inner_cell;
	    };

	    Kokkos::parallel_for("find_jf_ifft", fft_exec_policy, _find_jf_ifft );	    
	    //end fft
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);



	    //remove the average (for 1D problems only)
	    real_t jx_avg = 0,jy_avg=0,jz_avg=0;
	    
	    if(nx==1||ny==1||nz==1){
		if(nx>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfx(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jx_avg );
		    jx_avg /=n_inner_cell;
		}

		if(ny>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfy(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jy_avg );
		    jy_avg /=n_inner_cell;
		}

		if(nz>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfz(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jz_avg );
		    jz_avg /=n_inner_cell;
		}
	    }
	    
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
	    const real_t cjx = (nx>1)?dt_eps0:0;
	    const real_t cjy = (ny>1)?dt_eps0:0;
	    const real_t cjz = (nz>1)?dt_eps0:0;
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                ex(i) = ex(i) + ( - cjx * (jfx(i)-jx_avg) ) ;
                ey(i) = ey(i) + ( - cjy * (jfy(i)-jy_avg) ) ;
                ez(i) = ez(i) + ( - cjz * (jfz(i)-jz_avg) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "es_advance_e()" );
        }
    */
        void advance_e(
                field_array_t& fields,
                real_t,
                real_t,
                real_t,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

            serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
	    // printf("\n#after serial_update_ghosts: \n");
	    //for(int i=137; i<169; ++i) printf("%d %e\n",i,jfx(i));
	    // for (int x = 0; x < nx+2; x++) {
	    // 	for (int y = 0; y < ny+2; y++) {
	    // 	    for (int z = 0; z < nz+2; z++) {
	    // 		const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
	    // 		printf("%d, %d, %d, %d %e\n",x,y,z,i,jfx(i));
	    // 	    }
	    // 	}
	    // }
	    serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);
	    //printf("\n#after serial_update_ghosts_B: \n");
	    //for(int i=137; i<169; ++i) printf("%d %e\n",i,jfx(i));
	    // for (int x = 0; x < nx+2; x++) {
	    // 	for (int y = 0; y < ny+2; y++) {
	    // 	    for (int z = 0; z < nz+2; z++) {
	    // 		const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
	    // 		printf("%d, %d, %d, %d %e\n",x,y,z,i,jfx(i));
	    // 	    }
	    // 	}
	    // }

	    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
		{
		 const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
		 lsum += jfx(i);
		 //printf("xyz=%d,%d,%d,j=%e\n",x,y,z,jfx(i));
		};
	    real_t jx_avg;
	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});
	    Kokkos::parallel_reduce("find_jx_avg", fft_exec_policy, _find_javg, jx_avg );
	    jx_avg /=nx;

	    auto _find_jnorm = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
		{
		 const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
		 lsum += fabs(jfx(i));
		 //printf("xyz=%d,%d,%d,j=%e\n",x,y,z,jfx(i));
		};
	    real_t jx_norm;
	    Kokkos::parallel_reduce("find_jx_norm", fft_exec_policy, _find_jnorm, jx_norm );
	    jx_norm /=nx;

	    //printf("#jx_norm=%e\n",jx_norm);
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                const real_t cj = dt_eps0;
                ex(i) = ex(i) + ( - cj * (jfx(i))); // - jx_avg )) ;
                //ey(i) = ey(i) + ( - cj * jfy(i) ) ;
                //ez(i) = ez(i) + ( - cj * jfz(i) ) ;
		//printf("%d %e %e\n",i,jfx(i), ex(i));
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( "es_advance_e_1d()", exec_policy, _advance_e );

	    auto _find_eavg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
		{
		 const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
		 lsum +=  (jfx(i) - jx_avg); //ex(i); //
		 //printf("xyz=%d,%d,%d,j=%e\n",x,y,z,jfx(i));
		};
	    real_t ex_avg;
	    
	    //Kokkos::parallel_reduce("find_jx_avg", fft_exec_policy, _find_eavg, ex_avg );
	    //printf("#e_energy = %e\n",e_energy(fields, 0, 0, 0, nx, ny, nz, ng));
	    // for(int i=137; i<169; ++i) printf("%d %e\n",i,ex(i));
        }

    
    	  // Given E_n and E_{n+1/2}, puts E_{n+1} into the array of E_{n+1/2}

        void extend_e(
		  field_array_t& fields_nph, 
		  field_array_t& fields_n)
    {
            auto ex = Cabana::slice<FIELD_EX>(fields_nph);
            auto ey = Cabana::slice<FIELD_EY>(fields_nph);
            auto ez = Cabana::slice<FIELD_EZ>(fields_nph);
            
            auto ex_old = Cabana::slice<FIELD_EX>(fields_n);
            auto ey_old = Cabana::slice<FIELD_EY>(fields_n);
            auto ez_old = Cabana::slice<FIELD_EZ>(fields_n);
				
	    auto _extend_e = KOKKOS_LAMBDA( const int i )
            {
                ex(i) = ex(i)*2.0  - ex_old(i)  ;
                ey(i) = ey(i)*2.0  - ey_old(i)  ;
                ez(i) = ez(i)*2.0  - ez_old(i)  ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields_nph.size() );
            Kokkos::parallel_for( "extend_e()", exec_policy, _extend_e );
    }

};

// EM HERE: UNFINISHED
// TODO: Finish

class EM_Field_Solver
{
    public:

        //how to formalize/generalize this?

        real_t e_energy(
                field_array_t& fields,
                real_t,
                real_t,
                real_t,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto _e_energy = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
            {
                //lsum += ez(i)*ez(i);

                const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                lsum += ex(i)*ex(i) + ey(i)*ey(i) + ez(i)*ez(i);
            };

            real_t e_tot_energy=0;
            //Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            //Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({1,1,1}, {nx+1,ny+1,nz+1});
            Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
            real_t dV = 1.0; //Parameters::instance().dx * Parameters::instance().dy * Parameters::instance().dz;
            return e_tot_energy*0.5f*dV;
        }

        real_t b_energy(
                field_array_t& fields,
                real_t,
                real_t,
                real_t,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto _b_energy = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
            {
                const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                lsum += cbx(i)*cbx(i) + cby(i)*cby(i) + cbz(i)*cbz(i);
            };

            real_t b_tot_energy=0;
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({1,1,1}, {nx+1,ny+1,nz+1});
            Kokkos::parallel_reduce("b_energy", exec_policy, _b_energy, b_tot_energy );
            //TODO: no access to parameters here
            real_t dV = 1.0; //Parameters::instance().dx * Parameters::instance().dy * Parameters::instance().dz;
            return b_tot_energy*0.5f*dV;
        }


        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            //    auto ng = Parameters::instance().num_ghosts;
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);
            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);


            serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int x, const int y, const int z)
            {
                const real_t cj = dt_eps0;

                const int f0 = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                const int fx = VOXEL(x-1, y,   z,   nx, ny, nz, ng);
                const int fy = VOXEL(x,   y-1, z,   nx, ny, nz, ng);
                const int fz = VOXEL(x,   y,   z-1, nx, ny, nz, ng);

                ex(f0) = ex(f0) + ( - cj * jfx(f0) ) + ( py * (cbz(f0) - cbz(fy)) - pz * (cby(f0) - cby(fz)) );
                ey(f0) = ey(f0) + ( - cj * jfy(f0) ) + ( pz * (cbx(f0) - cbx(fz)) - px * (cbz(f0) - cbz(fx)) );
                ez(f0) = ez(f0) + ( - cj * jfz(f0) ) + ( px * (cby(f0) - cby(fx)) - py * (cbx(f0) - cbx(fy)) );

                //ex(f0) +=  ( - cj * jfx(f0) ) + ( py * (cbz(f0) - cbz(fy)) );

            };

            Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
            Kokkos::parallel_for( "advance_e()", zyx_policy, _advance_e );
        }


        void advance_b(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            //f0 = &f(x,  y,  z  );
            //fx = &f(x+1,y,  z  );
            //fy = &f(x,  y+1,z  );
            //fz = &f(x,  y,  z+1);

            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto _advance_b = KOKKOS_LAMBDA( const int x, const int y, const int z)
            {

                // Update value
                /*
                   f0->cbx -= ( py*( fy->ez-f0->ez ) - pz*( fz->ey-f0->ey ) );
                   f0->cby -= ( pz*( fz->ex-f0->ex ) - px*( fx->ez-f0->ez ) );
                   f0->cbz -= ( px*( fx->ey-f0->ey ) - py*( fy->ex-f0->ex ) );
                   */

                const int f0 = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                const int fx = VOXEL(x+1, y,   z,   nx, ny, nz, ng);
                const int fy = VOXEL(x,   y+1, z,   nx, ny, nz, ng);
                const int fz = VOXEL(x,   y,   z+1, nx, ny, nz, ng);

                cbx(f0) -= ( py*( ez(fy) - ez(f0) ) - pz*( ey(fz) - ey(f0) ) );
                cby(f0) -= ( pz*( ex(fz) - ex(f0) ) - px*( ez(fx) - ez(f0) ) );
                cbz(f0) -= ( px*( ey(fx) - ey(f0) ) - py*( ex(fy) - ex(f0) ) );


                //cbz(f0) -= - py*( ex(fy) - ex(f0) ) ;

            };

            Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
            Kokkos::parallel_for( "advance_b()", zyx_policy, _advance_b );
            serial_update_ghosts_B(cbx, cby, cbz, nx, ny, nz, ng);
        }
};

template<typename field_solver_t>
real_t dump_energies(
	std::vector<particle_list_t>& particles,	
        field_solver_t& field_solver,
        field_array_t& fields,
        int step,
        real_t time,
        real_t dx,
        real_t dy,
        real_t dz,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t ng,
	real_t k_tot_energy,
	real_t tot_en0=0	
        )
{
    //printf("%f\t",time);
    real_t e_en = field_solver.e_energy(fields, dx, dy, dz, nx, ny, nz, ng);


    // compute total kinetic energy
    real_t tot_energy,den;
    int num_sp = particles.size();

    tot_energy = e_en+k_tot_energy; //for ES
    if(step==0) den = 0;	
    else den = tot_energy - tot_en0;
    
    // Print energies to screen *and* dump them to disk
    // TODO: is it ok to keep opening and closing the file like this?
    // one per time step is probably fine?
    std::ofstream energy_file;

    if (step == 0)
    {
        // delete what is there
        energy_file.open("energies.txt", std::ofstream::out | std::ofstream::trunc);
#ifdef ES_FIELD_SOLVER	
	energy_file << "#ES field solver\n#step, time, E_field_energy, Kinetic energy, change of energy, relative energy error\n";
#else
	energy_file << "#ES field solver\n#step, time, E_field_energy, Kinetic energy, change of energy, B_field_energy\n";	
#endif	
	
    }
    else {
        energy_file.open("energies.txt", std::ios::app); // append
    }

    energy_file << step << " " << time << " " << e_en<<" "<<k_tot_energy<<" "<<den<<" "<<den/tot_en0;    
#ifndef ES_FIELD_SOLVER
    // Only write b info if it's available
    real_t b_en = field_solver.b_energy(fields, dx, dy, dz, nx, ny, nz, ng);
    energy_file << " " << b_en;
    printf("%d %f %e %e\n",step, time, e_en, b_en);
#else
    printf("%d %f %e %e\n",step, time, e_en,tot_energy);
    // printf("\n");
#endif
    energy_file << std::endl;
    energy_file.close();
    return tot_energy;    
}

#endif // pic_EM_fields_h
